import time
import random
from contextlib import suppress
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    CLIPImageProcessor,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from models.otter import OtterForConditionalGeneration, OtterConfig
from models.flamingo import FlamingoForConditionalGeneration
import os

try:
    from transformers.models.idefics.processing_idefics import image_attention_mask_for_packed_input_ids, incremental_to_binary_attention_mask
except ImportError:
    print("Failed to import Idefics processing module.")


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    
def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    return cast_dtype


def get_autocast(precision):
    if precision == "amp":
        return torch.cuda.amp.autocast
    elif precision == "amp_bfloat16" or precision == "amp_bf16":
        # amp_bfloat16 is more stable than amp float16 for clip training
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    elif precision == "fp16":
        return lambda: torch.cuda.amp.autocast(dtype=torch.float16)
    else:
        return suppress


def get_checkpoint(model):
    state_dict = model.state_dict()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            del state_dict[name]

    return state_dict


def get_checkpoint_deepspeed_zero3(args, model):
    state_dict = {}

    for name, p in model.named_parameters():
        if p.requires_grad:
            state_dict[name] = p.data
    return state_dict


def get_model(args, accelerator, device_id):
    if args.pretrained_model_name_or_path is not None:
        accelerator.print(f"Loading pretrained model from {args.pretrained_model_name_or_path}")
        device_map = {"": device_id} if accelerator.distributed_type == "MULTI_GPU" or accelerator.distributed_type == "DEEPSPEED" else "auto"
        kwargs = {"local_files_only": args.offline, "device_map": device_map}
        
        if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
            kwargs.pop("device_map")
        if args.customized_config is not None:
            kwargs["config"] = args.customized_config
        
        if "otter" in args.model_name.lower():
            config = OtterConfig.from_pretrained(args.pretrained_model_name_or_path)
            config.update({'max_num_frames': args.max_num_frames})
            print('=====> Begin loading model <=========')
            model = OtterForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                config=config,
                local_files_only=True,
            )
            print('=====> Done loading model <=========')
            
            ### important: change padding side if testing ###
            if args.testonly:
                model.text_tokenizer.padding_side='left'

            args.tokenizer = model.text_tokenizer
            tokenizer = model.text_tokenizer
            image_processor = CLIPImageProcessor()

        elif "flamingo" in args.model_name.lower():
            model = FlamingoForConditionalGeneration.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            ### important: change padding side if testing ###
            if args.testonly:
                model.text_tokenizer.padding_side='left'

            # add special tokens for instruction tuning
            model.text_tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"]})
            args.tokenizer = model.text_tokenizer
            tokenizer = model.text_tokenizer
            image_processor = CLIPImageProcessor()
        
        elif "idefics" in args.model_name.lower():
            model = IdeficsForVisionText2Text.from_pretrained(
                args.pretrained_model_name_or_path,
                **kwargs,
            )
            if args.gradient_checkpointing:
                model.gradient_checkpointing_enable()

            if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
                params_to_gather = [p for name, p in model.named_parameters() if p.requires_grad]
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        print(device_id, f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B",)
            else:
                print(device_id, f"IDEFICS Trainable Params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)) / 1e9:.3f} B",)
            
            processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, legacy=False)
            past_special_tokens = processor.tokenizer.special_tokens_map["additional_special_tokens"]
            processor.tokenizer.add_special_tokens({"additional_special_tokens": ["<answer>"] + past_special_tokens})
            
            if args.testonly:
                processor.tokenizer.padding_side='left'
            
            image_processor = processor.image_processor
            tokenizer = processor.tokenizer
            # make embedding size divisible by 64 for hardware compatiblity https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
            new_embedding_size = (len(tokenizer) // 64 + 1) * 64
            model.resize_token_embeddings(new_embedding_size, pad_to_multiple_of=64)
            
    else:
        print('Please specify a pretrained-model-named-path')
        raise NotImplementedError
    
    ### load trained checkpoint, if necessary, mainly for testing ###
    if args.trained_ckpt is not None:
        print('Begin loading ckpt from: ', args.trained_ckpt)
        train_ckpt = torch.load(args.trained_ckpt, map_location="cpu")
        if train_ckpt.get("model_state_dict", None) is not None:
            train_ckpt = train_ckpt["model_state_dict"]

        res = model.load_state_dict(train_ckpt, strict=False)
        print(res)
    
    ### resize token embeddings, if necessary ###    
    if hasattr(model, "lang_encoder") and "LlamaForCausalLM" in model.lang_encoder.__class__.__name__:
        model.lang_encoder.resize_token_embeddings(len(model.text_tokenizer))

    return model, tokenizer, image_processor


def resume_from_checkpoint(args, model, optimizer, lr_scheduler):
    resume_from_epoch = 0
    args.external_save_dir = os.path.join(args.external_save_dir, args.run_name) if args.external_save_dir else args.run_name
    
    if os.path.exists(f"{args.external_save_dir}") and args.resume_from_checkpoint is True:
        checkpoint_list = glob.glob(f"{args.external_save_dir}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.external_save_dir}.")
        else:
            resume_from_checkpoint_path = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"Found checkpoint {resume_from_checkpoint_path} for run {args.external_save_dir}.")

        if args.rank == 0:
            print(f"Loading checkpoint from {resume_from_checkpoint_path}")
            
        checkpoint = torch.load(resume_from_checkpoint_path, map_location="cpu")
        res = model.load_state_dict(checkpoint["model_state_dict"], False)
        print('Resuming checkpoint status: ', res)
        
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1
    
    return resume_from_epoch

def get_grouped_params(args, model):
    params_with_wd, params_without_wd = [], []

    def apply_decay(x):
        return "gated_cross_attn_layer" in x and "ff_gate" not in x and "attn_gate" not in x and "norm" not in x and "bias" not in x

    for n, p in model.named_parameters():
        # if p.requires_grad:
        if apply_decay(n):
            params_with_wd.append(p)
        else:
            params_without_wd.append(p)

    return [
        {"params": params_with_wd, "weight_decay": args.weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def get_optimizer_and_lr_scheduler(args, loader, model):    
    total_training_steps = len(loader) * args.num_epochs

    optimizer = torch.optim.AdamW(get_grouped_params(args, model), lr=args.learning_rate)

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    args.warmup_steps = total_training_steps * args.warmup_steps_ratio if args.warmup_steps_ratio is not None else args.warmup_stepsps

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps // args.gradient_accumulation_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    return optimizer, lr_scheduler


def save_checkpoint(args, accelerator, model, optimizer, lr_scheduler, epoch):
    if args.rank == 0:
        if not os.path.exists(args.external_save_dir):
            os.makedirs(args.external_save_dir)

    if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
        checkpoint_dict = accelerator.get_state_dict(model)

        if args.rank == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
            for name in list(checkpoint_dict.keys()):
                if name not in trainable_params_name:
                    del checkpoint_dict[name]

    else:
        if args.rank == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(unwrapped_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            }
            # checkpoint_dict = {
            #     "model_state_dict": get_checkpoint(unwrapped_model),
            # }

    if args.rank == 0:
        print(f"Saving checkpoint to {args.external_save_dir}/checkpoint_{epoch}.pt")
        accelerator.save(checkpoint_dict, f"{args.external_save_dir}/checkpoint_{epoch}.pt")
        # save the config
        unwrapped_model.config.save_pretrained(args.external_save_dir)
        if args.delete_previous_checkpoint:
            if epoch > 0:
                os.remove(f"{args.external_save_dir}/checkpoint_{epoch-1}.pt")


def save_final_checkpoint(args, accelerator, model, optimizer, lr_scheduler, epoch):
    if args.rank == 0:
        if not os.path.exists(args.external_save_dir):
            os.makedirs(args.external_save_dir)

    if accelerator.distributed_type == "DEEPSPEED" and accelerator.state.deepspeed_plugin.zero_stage == 3:
        checkpoint_dict = accelerator.get_state_dict(model)

        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.config.save_pretrained(args.external_save_dir)

        if args.rank == 0 and not args.save_hf_model:
            trainable_params_name = [name for name, p in unwrapped_model.named_parameters() if p.requires_grad]
            for name in list(checkpoint_dict.keys()):
                if name not in trainable_params_name:
                    del checkpoint_dict[name]

            accelerator.save(
                checkpoint_dict,
                f"{args.external_save_dir}/final_weights.pt",
            )
        elif args.rank == 0 and args.save_hf_model:
            unwrapped_model.save_pretrained(
                f"{args.external_save_dir}",
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=checkpoint_dict,
            )

    else:
        if args.rank == 0:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint_dict = get_checkpoint(model=unwrapped_model)

            accelerator.save(
                checkpoint_dict,
                f"{args.external_save_dir}/final_weights.pt",
            )
            # save the config
            unwrapped_model.config.save_pretrained(args.external_save_dir)

            if args.report_to_wandb and args.save_checkpoints_to_wandb:
                wandb.save(f"{args.external_save_dir}/final_weights.pt")
            if args.save_hf_model:
                unwrapped_model.save_pretrained(f"{args.external_save_dir}")

def visualize_data(images):    
    FLAMINGO_MEAN = [0.481, 0.458, 0.408]
    FLAMINGO_STD = [0.269, 0.261, 0.276]

    FLAMINGO_MEAN_INT = [x * 255 for x in FLAMINGO_MEAN]
    FLAMINGO_STD_INT = [x * 255 for x in FLAMINGO_STD]
    
    ## [b, T_img, F, C, H, W]
    image = images[0][0][0]
    image[0] = image[0] * FLAMINGO_STD_INT[0] + FLAMINGO_MEAN_INT[0]
    image[1] = image[1] * FLAMINGO_STD_INT[1] + FLAMINGO_MEAN_INT[1]
    image[2] = image[2] * FLAMINGO_STD_INT[2] + FLAMINGO_MEAN_INT[2]
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('demo/debug_image.jpg', image)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DistributedProxySampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of input sampler indices.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Input sampler is assumed to be of constant size.

    Arguments:
        sampler: Input data sampler.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, sampler, num_replicas=None, rank=None):
        super(DistributedProxySampler, self).__init__(sampler, num_replicas=num_replicas, rank=rank, shuffle=False)
        self.sampler = sampler

    def __iter__(self):
        # deterministically shuffle based on epoch
        torch.manual_seed(self.epoch)
        indices = list(self.sampler)

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        if len(indices) != self.total_size:
            raise RuntimeError("{} vs {}".format(len(indices), self.total_size))

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("{} vs {}".format(len(indices), self.num_samples))

        return iter(indices)


# supporting idefics processing
def get_image_attention_mask(output_input_ids, max_num_images, tokenizer, include_image=True):
    # image_attention_mask, _ = image_attention_mask_for_packed_input_ids(output_input_ids, tokenizer)
    # image_attention_mask = incremental_to_binary_attention_mask(image_attention_mask, num_classes=max_num_images)
    if include_image:
        image_attention_mask, _ = image_attention_mask_for_packed_input_ids(output_input_ids, tokenizer)
        image_attention_mask = incremental_to_binary_attention_mask(image_attention_mask, num_classes=max_num_images)
    else:
        # in full language mode we set the image mask to all-0s
        image_attention_mask = torch.zeros(output_input_ids.shape[0], output_input_ids.shape[1], 1, dtype=torch.bool)
    return image_attention_mask
