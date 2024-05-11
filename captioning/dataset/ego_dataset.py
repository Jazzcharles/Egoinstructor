# Copyright 2023 The Otter Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import base64
from io import BytesIO
import re
import contextlib
import os
import orjson
# import ijson.backends.yajl2_cffi as ijson
from PIL import ImageFile
from torchvision import transforms
import random

import sys
from PIL import Image, ImageFile

import torch
import numpy as np
import cv2

from ipdb import set_trace
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

import torchvision.transforms._transforms_video as transforms_video
from .data_utils import video_loader, Permute

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

FLAMINGO_MEAN = [0.481, 0.458, 0.408]
FLAMINGO_STD = [0.269, 0.261, 0.276]

FLAMINGO_MEAN_INT = [x * 255 for x in FLAMINGO_MEAN]
FLAMINGO_STD_INT = [x * 255 for x in FLAMINGO_STD]

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def random_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    numpy_state = np.random.get_state()
    random_state = random.getstate()
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(random_state)


class EgoDataset(Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.tokenizer = args.tokenizer
        self.is_testing = args.testonly
        self.clean_narration = args.clean_narration  
        self.xview = args.xview
        self.use_chat = args.use_chat
        self.max_shot = args.max_shot
        self.load_from = args.load_from
        self.chunk_len_ego = 300
        self.chunk_len_exo = -1
        self.fps_ego = 30
        self.fps_exo = -1
        
        print('$' * 100)
        print('Whether to use clean narration: ', self.clean_narration)
        print('Whether to use chatdata:', self.use_chat)
        print('Whether to use xview: ', self.xview)
        print('Whether to use fewshot:', self.max_shot)
        print('Way to load the data:', self.load_from)
        print('$' * 100)
        
        self.seed = args.seed
        self.patch_image_size = args.patch_image_size
        self.max_seq_len = args.max_seq_len

        self.epoch = 0

        self.inst_format = args.inst_format
        self.resample_frames = args.resample_frames
        self.wrap_sys = f"<<SYS>>\nYou are a helpful vision language assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.\n<</SYS>>\n\n"

        scales = [(args.patch_image_size, args.patch_image_size)]

        self.video_resize_transform = transforms.Compose(
            [
                Permute([3, 0, 1, 2]),  # T H W C -> C T H W
                transforms.Resize((args.patch_image_size, args.patch_image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms_video.NormalizeVideo(mean=FLAMINGO_MEAN_INT, std=FLAMINGO_STD_INT),
            ]
        )
        
        ### prepare metadata ###
        assert args.datapath_ego != "", f"Error: The datapath for ego videos do not get!"
        assert args.datapath_exo != "", f"Error: The datapath for exo videos do not get!"
        assert args.metapath != "", f"Error: The metapath do not get!"
        assert args.train_config_path != "", f"Error: The train config path do not get!"

        self.datapath_ego = args.datapath_ego
        self.datapath_exo = args.datapath_exo
        self.metapath = args.metapath
        self.train_config_path = args.train_config_path

        if self.load_from == 'ceph':
            from petrel_client.client import Client 
            self.client = Client()
        else:
            self.client = None

        self.dataset = {}
        self.images = {}
        self.train_data_list = []
        self.train_config = []
        self.task_name = args.task_name

        with open(self.metapath, "rb") as f:
            self.dataset = orjson.loads(f.read())["data"]
        
        with open(self.train_config_path, "rb") as f:
            self.train_config = orjson.loads(f.read())

        self.train_data_list = list(self.train_config.keys())

        self.bos_item = torch.LongTensor([args.tokenizer.bos_token_id]) if args.tokenizer is not None else None
        self.eos_item = torch.LongTensor([args.tokenizer.eos_token_id]) if args.tokenizer is not None else None
        self.bos_mask = torch.LongTensor([1])
        self.eos_mask = torch.LongTensor([1])

    def load_single_video(self, ins, root, chunk_len=-1, fps=-1, clip_length=32):
        sample = ins['image_ids'][0]
        video_id = sample['vid']
        start_sec = float(sample['start_second'])
        end_sec = float(sample['end_second'])
        patch_images = video_loader(root, video_id, start_sec, end_sec,
            chunk_len=chunk_len, fps=fps, clip_length=clip_length,
        )
        # F, H, W, C
        patch_images = self.video_resize_transform(patch_images)
        # T_img, F, C, H, W
        patch_images = patch_images.transpose(1, 0).unsqueeze(0)
        return patch_images

    def random_init_case(self, question):
        if len(question) == 0:
            return question

        first_letter = question[0]
        if random.choice([True, False]):
            first_letter = first_letter.upper()
        else:
            first_letter = first_letter.lower()

        return first_letter + question[1:]

    def narration_filter(self, x):
        if x in ['#' , 'c' , 'cc', 'o' , 'x', 'y', 'b', 'p', 's', 'r', 'g', 'n', 'z', 'v', 'k']:
            return ''
        return x

    def clear_narration(self, narration):       
        if self.clean_narration: 
            alltext = word_tokenize(narration.lower())
            filtered_text = [self.narration_filter(x) for x in alltext]
            filtered_text = [x for x in filtered_text if len(x)]
            narration = ' '.join(filtered_text)
            return narration
        else:
            return narration

    def pre_question(self, question):
        question = question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " ")
        question = self.random_init_case(question)

        question = re.sub(
            r"\s{2,}",
            " ",
            question,
        )
        question = question.lstrip("\n")
        question = question.rstrip("\n")
        question = question.strip(" ")

        return question

    def pre_answer(self, answer, max_ans_words=1024):
        answer = re.sub(
            r"\s{2,}",
            " ",
            answer,
        )
        answer = answer.rstrip("\n")
        answer = answer.strip(" ")

        # truncate question
        return_answer = ""
        answers = answer.split(".")

        for _ in answers:
            if return_answer == "":
                cur_answer = _
            else:
                cur_answer = ".".join([return_answer, _])
            if len(cur_answer.split(" ")) <= max_ans_words:
                return_answer = cur_answer
            else:
                break

        if return_answer == "":
            answer_words = answer.split(" ")
            return_answer = " ".join(answer_words[:max_ans_words])
        else:
            if return_answer[-1] != "." and return_answer != answers:
                return_answer += "."

        return return_answer


    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def process_online_videoqa(self, instruction_id, instruction, answer, image_ids, in_context_example_ids, resample_frames=32, inst_format="simple"):
        patch_images = torch.tensor([])
        all_texts = ""
        
        def prepare_text(cur_instruction_id, is_last=False):
            ### build text instruction ###
            cur_instruction = self.dataset[cur_instruction_id]["instruction"]
            cur_instruction = self.pre_question(cur_instruction)
            if self.use_chat:
                cur_answer = self.dataset[cur_instruction_id]["chat_answer"]
            else:
                cur_answer = self.dataset[cur_instruction_id]["answer"]

            cur_answer = self.pre_answer(cur_answer)
            if inst_format == "llama2":
                if is_last is False:
                    cur_text = f"[INST]{self.wrap_sys}<image>{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"[INST]{cur_instruction}[/INST]<answer>{cur_answer}<|endofchunk|>"
            
            elif inst_format == "idefics":
                if is_last is False:
                    cur_text = f"User:<fake_token_around_image><image><fake_token_around_image>{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
                else:
                    cur_text = f"User:{cur_instruction} Assistant:<answer>{cur_answer}<|endofchunk|>"
            
            elif inst_format == "simple":
                ### prepare the answer ###
                if self.is_testing is False:
                    ### training, we always have such format ###
                    cur_text = f"<image>User:{cur_instruction} GPT:<answer>{self.clear_narration(cur_answer)}<|endofchunk|>"
                else:
                    ### testing, for few-shot support samples, we have their answers ###
                    if is_last is False:
                        cur_text = f"<image>User:{cur_instruction} GPT:<answer>{self.clear_narration(cur_answer)}<|endofchunk|>"
                    else:
                    ### last case, i.e. the real testing case, we do not have answer ###
                        cur_text = f"<image>User:{cur_instruction} GPT:<answer>"
        
            return cur_text
        
        ### load query image, but should be appended after in-context samples ###
        meta = image_ids[0]
        query_image = video_loader(self.client, self.datapath_ego, meta['vid'], float(meta['start_second']), float(meta['end_second']), chunk_len=self.chunk_len_ego, fps=self.fps_ego, clip_length=resample_frames)
        query_image = self.video_resize_transform(query_image)
        query_image = query_image.transpose(1, 0).unsqueeze(0)
        
        if not isinstance(in_context_example_ids, list):
            in_context_example_ids = [in_context_example_ids]

        ### load in-context samples ###
        if self.xview and self.max_shot > 0:
            in_context_images = []
            success_instruction_ids = []

            ### 1. load support samples ####
            for idx, cur_instruction_id in enumerate(in_context_example_ids[:]):
                ### load meta ###
                cur_sample = self.dataset[cur_instruction_id]

                ### load text first ###
                cur_text = prepare_text(cur_instruction_id, is_last=False)
                all_texts += cur_text

                ### load videos ###
                try:
                    exo_video = self.load_single_video(cur_sample, root=self.datapath_exo, chunk_len=self.chunk_len_exo, fps=self.fps_exo, clip_length=resample_frames)
                    success_instruction_ids.append(cur_instruction_id)
                except:
                    ### load current video failed, try next support sample ### 
                    continue
                
                in_context_images.append(exo_video)
                if len(in_context_images) == self.max_shot:
                    break

            if len(in_context_images) < self.max_shot:
                if len(in_context_images) == 0:
                    ### all support failed, load query image, and all support texts ###
                    in_context_images = [query_image for shot_id in range(self.max_shot)]
                    for shot_id, temp_id in enumerate(in_context_example_ids[:]):
                        all_texts += prepare_text(temp_id, is_last=False)
                        if shot_id == self.max_shot - 1:
                            break
                else:
                    ### we randomly select from succeed ones ###
                    lack = self.max_shot - len(in_context_images)
                    selected_ids = random.choices(success_instruction_ids, k=lack)
                    for each_id in selected_ids:
                        ### load videos ###
                        selected_sample = self.dataset[each_id]
                        selected_video = self.load_single_video(selected_sample, root=self.datapath_exo, chunk_len=self.chunk_len_exo, fps=self.fps_exo, clip_length=resample_frames)
                        in_context_images.append(selected_video)

                        selected_text = prepare_text(each_id, is_last=False)
                        all_texts += selected_text
                
            context_images = torch.cat(in_context_images, dim=0)
            patch_images = torch.cat([context_images, query_image], dim=0)

        elif self.xview and self.max_shot == 0:
            ### we just randomly pick another sample as format guidance ###
            ### For simplicity, just use the 0-th or 1-st one ###
            cur_id = instruction_id.split('_')[-1]
            support_id = '000000' if cur_id != '000000' else '000001'
            full_support_id = instruction_id.replace(cur_id, support_id)
            all_texts += prepare_text(full_support_id, is_last=False)

            ### keep the image ###
            patch_images = query_image
        else:
            patch_images = query_image
        
        ### ADD the query sample ###
        all_texts += prepare_text(instruction_id, is_last=True)
        
        # print(patch_image.shape, all_texts)
        # set_trace()
        return patch_images, all_texts


    def process_image_text_pair(self, index):
        # try:
        cur_train_id = self.train_data_list[index]
        
        (
            instruction_id,
            instruction,
            answer,
            image_ids,
            in_context_example_ids,
        ) = (
            cur_train_id,
            self.dataset[cur_train_id]["instruction"],
            self.dataset[cur_train_id]["answer"],
            self.dataset[cur_train_id]["image_ids"],
            self.train_config[cur_train_id],
        )
        inst_format = self.inst_format
        resample_frames = self.resample_frames
        
        patch_images, all_texts = self.process_online_videoqa(
            instruction_id, instruction, answer, image_ids, in_context_example_ids, resample_frames=resample_frames, inst_format=inst_format
        )
        
        all_text = self.tokenizer(
            f"{all_texts}",
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_seq_len,  # for current 2k mpt/llama model, setting to 2048 causes error (2042 works)
        )

        all_item = all_text["input_ids"].squeeze(0)
        all_item_mask = all_text["attention_mask"].squeeze(0)

        all_item = torch.cat([self.bos_item, all_item, self.eos_item])
        all_item_mask = torch.cat([self.bos_mask, all_item_mask, self.eos_mask])
        
        example = {
            "id": instruction_id,
            "source": all_item,
            "text_mask": all_item_mask,
            "patch_images": patch_images,
            "answer": self.clear_narration(answer),
            'query': all_texts,
        }

        return example

    def __str__(self):
        return f"type: {type(self)}, length: {len(self)}"

    def __len__(self):
        return len(self.train_data_list)

    def __getitem__(self, index):
        with random_seed(self.seed, self.epoch):
            pair_sample = self.process_image_text_pair(index)
            # if dataset is not supported
            if pair_sample is None:
                return self.__getitem__(index + 1)
        return pair_sample

    def collate(self, samples):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []  # containing image-text pairs
        for sample_tuple in samples:
            samples_v1.append(sample_tuple)

        res_v1 = collate_fn(
            samples_v1,
            pad_idx=self.tokenizer.pad_token_id,
            eos_idx=self.tokenizer.eos_token_id,
        )
        return res_v1


def collate_fn(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key, pad_idx, pading_size=None):
        res = collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
            pad_to_length=pading_size,
        )
        return res

    larger_size = max([s["source"].size(0) for s in samples])

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source", pad_idx=pad_idx, pading_size=larger_size)
    src_tokens_masks = merge("text_mask", pad_idx=0, pading_size=larger_size)
    answer = [s["answer"] for s in samples]
    query = [s['query'] for s in samples]

    batch = {
        "id": id,
        "nsentences": len(samples),
        "net_input": {
            "input_ids": src_tokens,
            "attention_masks": src_tokens_masks,
        },
        "answer": answer,
        "query": query,
    }
    larger_incontext_num = max([s["patch_images"].size(0) for s in samples])
    if samples[0].get("patch_images", None) is not None:
        batch["net_input"]["patch_images"] = torch.stack([sample["patch_images"] for sample in samples], dim=0)

    return batch


def collate_tokens(
    values,
    pad_idx,
    eos_idx=None,
    left_pad=False,
    move_eos_to_beginning=False,
    pad_to_length=None,
    pad_to_multiple=1,
    pad_to_bsz=None,
):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    size = size if pad_to_length is None else max(size, pad_to_length)
    if pad_to_multiple != 1 and size % pad_to_multiple != 0:
        size = int(((size - 0.1) // pad_to_multiple + 1) * pad_to_multiple)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            if eos_idx is None:
                # if no eos_idx is specified, then use the last token in src
                dst[0] = src[-1]
            else:
                dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    if values[0].dim() == 1:
        res = values[0].new(len(values), size).fill_(pad_idx)
    elif values[0].dim() == 2:
        assert move_eos_to_beginning is False
        res = values[0].new(len(values), size, values[0].size(1)).fill_(pad_idx)
    else:
        raise NotImplementedError

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v) :] if left_pad else res[i][: len(v)])
    return res

