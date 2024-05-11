# Training and evaluation data

### Ego4d videos
- Download the official ego4d videos from the official [Ego4d website](https://ego4d-data.org/docs/start-here/#cli-download).
- To enable faster loading, we use the scripts provided by [EgoVLP](https://github.com/showlab/EgoVLP) to resize the videos with a shorter side of 320.
```
python utils/video_resize.py
```
and chunk each video into several 5-minute (300 secs) video clips by running
```
python utils/video_chunk.py
```

### HowTo100M videos 
- Prepare and download HowTo100M videos from [the official website](https://www.di.ens.fr/willow/research/howto100m/).

### Metadata used for training and evaluation ###
We provide the processed train/val metadata file, train/val config file at the huggingface dataset.

| File | Link |
|-------------------------|--------|
| E4DOL_crossview_train_instructions.json | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/E4DOL_crossview_train_instructions.json)
| E4DOL_crossview_train_config.json | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/E4DOL_crossview_train_config.json)
| E4DOL_crossview_val_instructions.json | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/E4DOL_crossview_val_instructions.json)
| E4DOL_crossview_val_config.json | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/E4DOL_crossview_val_config.json)

For the train metadata file, the structure of each element is as follows:
```
"E4DOL_INS_000000": {
    "instruction": "Describe the main action in the video.",
    "answer": "#C C carries a pot from the cooker",
    "chat_answer": "#C C carries a pot from the cooker",
    "image_ids": [{
        "vid": "9e4edf4d-e557-4b3d-bc35-0d7f1f91019b",
        "uid": "df6a908a-5c71-11ee-a42a-80615f168c41",
        "start_second": 3.8109597588600206,
        "end_second": 4.440017441139981,
    }],
    "rel_ins_ids": [
        "E4DOL_INS_000001", ..., "E4DOL_INS_000015",
    ]
}
```

Here, for each egocentric video, it has video id, start/end timestamp, narration (answer) and the a list of keys regarding the few-shot support exo videos (rel_in_ids). i.e., E4DOL_INS_000001, ..., E4DOL_INS_000015 are 15 retrieved relevant exocentric videos from HowTo100M.

For the train config file, the structure of each element is as follows:
```
"E4DOL_INS_000000": 
    ["E4DOL_INS_000001", ..., "E4DOL_INS_000015"]
"E4DOL_INS_000016": 
    ["E4DOL_INS_000017", ..., "E4DOL_INS_000031"]
```
The keys refer to egocentric videos that are used for training, and the values are few-shot exocentric videos.


