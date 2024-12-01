import mimetypes
import os
from typing import Union
import cv2
import requests
import torch
import transformers
from PIL import Image
import argparse
import decord
import sys
import json
import numpy as np

# make sure you can properly access the otter folder
from models.otter import OtterForConditionalGeneration, OtterConfig

from accelerate import Accelerator
from transformers import CLIPImageProcessor
from torchvision import transforms
import torchvision.transforms._transforms_video as transforms_video

# Disable warnings
requests.packages.urllib3.disable_warnings()

# ------------------- Utility Functions -------------------
FLAMINGO_MEAN = [0.481, 0.458, 0.408]
FLAMINGO_STD = [0.269, 0.261, 0.276]
FLAMINGO_MEAN_INT = [x * 255 for x in FLAMINGO_MEAN]
FLAMINGO_STD_INT = [x * 255 for x in FLAMINGO_STD]
video_resize_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms_video.NormalizeVideo(mean=FLAMINGO_MEAN_INT, std=FLAMINGO_STD_INT),
])


def post_process_generation(cur_str):
    cur_str = cur_str.split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
    return cur_str

def get_response(
    video_array, prompt: str, 
    accelerator, model=None, tokenizer=None,
    tensor_dtype=None,
) -> str:
    endofchunk_text = (
        "<|endofchunk|>" if "<|endofchunk|>" in tokenizer.special_tokens_map["additional_special_tokens"] else "<end_of_utterance>"
    )  # for different tokenizer
    endofchunk_token_id = tokenizer(endofchunk_text, add_special_tokens=False)["input_ids"][-1]
    bad_words_id = tokenizer(["User:", "GPT1:", "GFT:", "GPT:"], add_special_tokens=False).input_ids
    all_text = tokenizer(
        f"{prompt}",
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=2000,  
        # for current 2k mpt/llama model, setting to 2048 causes error (2042 works)
    )
    input_ids = all_text["input_ids"] #.squeeze(0)
    attention_mask = all_text["attention_mask"] #.squeeze(0)
    
    # input_ids = input_ids[:, :-1]
    # attention_mask = attention_mask[:, :-1]

    generated_text = model.generate(
        vision_x=video_array.to(model.device, dtype=tensor_dtype).unsqueeze(0), #[B, t_img, T, C, H, W]
        lang_x=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=64,
        num_beams=3,
        no_repeat_ngram_size=3,
        bad_words_ids=bad_words_id,
    )

    all_pred = accelerator.pad_across_processes(generated_text, dim=1, pad_index=endofchunk_token_id)
    parsed_output = tokenizer.batch_decode(all_pred)[0]
    out = parsed_output.split("<answer>")[-1].lstrip().rstrip().split("<|endofchunk|>")[0].lstrip().rstrip().lstrip('"').rstrip('"')
    return out

def load_video(video_path, video_length):
    video = decord.VideoReader(video_path)
    indices = np.linspace(0, len(video) - 1, video_length, dtype=int)
    
    video_array = video.get_batch(indices).asnumpy()
    video_array = torch.tensor(video_array).permute(3, 0, 1, 2) # T H W C -> C T H W
    video_array = video_resize_transform(video_array.float())
    video_array = video_array.transpose(1, 0).unsqueeze(0) #[B, T, C, H, W]
    return video_array

def process_text(
    text, instruction="Describe the content of the video", is_last=False,
):
    if is_last is False:
        new_text = f"<image>User:{instruction} GPT:<answer>{text}<|endofchunk|>"
    else:
        ### last case, i.e. the real testing case, we do not have answer ###
        new_text = f"<image>User:{instruction} GPT:<answer>"
    return new_text

def inference(args):
    # ------------------- Main Function -------------------
    load_bit = "fp32"
    if load_bit == "fp16":
        precision = {"torch_dtype": torch.float16}
    elif load_bit == "bf16":
        precision = {"torch_dtype": torch.bfloat16}
    elif load_bit == "fp32":
        precision = {"torch_dtype": torch.float32}
    
    accelerator = Accelerator(mixed_precision='no' if load_bit == 'fp32' else load_bit)
    device_id = accelerator.device
    
    config = OtterConfig.from_pretrained(args.pretrained_name_or_path)
    config.update({'max_num_frames': args.max_num_frames})
    print('=====> Begin loading model <=========')
    model = OtterForConditionalGeneration.from_pretrained(
        args.pretrained_name_or_path,
        config=config,
        local_files_only=True, ### feel free to change this if you can directly access to huggingface
        #**precision,
    )
    print('=====> Done loading model <=========')
    
    ### important: change padding side while testing ###
    model.text_tokenizer.padding_side='left'    
    tokenizer = model.text_tokenizer
    tensor_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[load_bit]
    model.eval()
    
    
    ### load pretrained checkpoint ###
    ckpt = torch.load(args.pretrained_checkpoint)
    if 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']
    res = model.load_state_dict(ckpt, strict=False)
    print(res)
    # breakpoint()
    
    ### prepare data and begin inference ###
    testdata = json.load(open(args.testdata))
    print(f'Processing {len(testdata)} samples')
    for eachdata in testdata:
        ego_video_path = eachdata['ego_video']
        ego_video = load_video(ego_video_path, args.max_num_frames)
        
        all_videos = []
        all_captions = ""
        
        ### process in-context samples ###
        for i in range(args.max_shot):
            exo_video_path = eachdata['exo_video'][i]['path'] 
            exo_video_caption = eachdata['exo_video'][i]['caption'] 
            exo_video = load_video(exo_video_path, args.max_num_frames)
            
            all_videos.append(exo_video)
            all_captions += process_text(exo_video_caption, is_last=False)
    

        ### process last ego sample ###
        all_videos = torch.cat(all_videos, dim=0)
        
        all_videos = torch.cat([ego_video, all_videos], dim=0)
        all_captions += process_text("", is_last=True)

        print(f'Done preparing video with shape {all_videos.shape}')
        print(f'Prompt is: {all_captions}')
        
        ### begin inference ###    
        response = get_response(all_videos, all_captions, accelerator, model, tokenizer, tensor_dtype)
        print(f"Response: {response}")

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main inference script for the model")
    parser.add_argument('--testdata', default='./assets/testdata.json', type=str)
    parser.add_argument('--max_shot', default=4, type=int)
    parser.add_argument('--max_num_frames', default=32, type=int)
    parser.add_argument('--pretrained_name_or_path', default='luodian/OTTER-MPT1B-RPJama-Init', type=str)
    # parser.add_argument('--pretrained_name_or_path', default='/mnt/petrelfs/xujilan/.cache/huggingface/hub/models--luodian--OTTER-MPT1B-RPJama-Init/snapshots/74490d8a17c2db46290a19b33229c7a2c62a8528/', type=str)
    parser.add_argument('--pretrained_checkpoint', default='/mnt/petrelfs/xujilan/tools/Otter/checkpoints/OTTER-MPT1B-RPJama-xview-4shot-new-8gpu/checkpoint_0.pt', type=str)
    args = parser.parse_args()
    inference(args)