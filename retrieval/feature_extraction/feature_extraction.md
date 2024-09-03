# Feature extraction

### Prepare metadata info 
1. Put all the videos under one folder
```
- video_root:
    - xxx.mp4
    - yyy.mp4
```
2. Prepare video_info.csv that contains the video name and duration (seconds)
```
xxx.mp4 123.2
yyy.mp4 553.7
```

### Ego video features
```
python main_feature_extraction.py \
    --model videomae \
    --ego_checkpoint /path/to/your/videomae_L14_checkpoint_best.pth \
    --video_root /path/to/videos \
    --output_dir /path/to/save_your_videos/ \
    --video_csv /path/to/your/video_info.csv \
```

### Exo video features 
1. Clone the [InternVideo](https://github.com/OpenGVLab/InternVideo/tree/main) repo and copy InternVideo1 to ./models
```
git clone https://github.com/OpenGVLab/InternVideo.git
cp -r InternVideo/InternVideo1 models/InternVideo
```

2. Extract exo-video features
```
python main_feature_extraction.py \
    --model internvideo-ffmpeg \
    --ego_checkpoint /path/to/your/InternVideo-MM-L-14.ckpt \
    --video_root /path/to/videos \
    --output_dir /path/to/save_your_videos/ \
    --video_csv /path/to/your/video_info.csv \
```

### Ego/Exo-video encoders
| Pretrained Model | Link | Size |
|-------------------------|--------|--------|
| VideoMAE-L (Ego) | 🤗 [HF link](https://huggingface.co/Jazzcharles/EgoInstructor-ModelZoo/resolve/main/videomae_L14_checkpoint_best.pth) | 5.1GB
| InternVideo-L (Exo) | 🤗 [HF link](https://huggingface.co/OpenGVLab/InternVideo1.0/resolve/main/InternVideo-MM-L-14.ckpt) | 2.47GB

### Acknowledgement
This script is modified from [TAN](https://github.com/TengdaHan/TemporalAlignNet/tree/main/htm_zoo). Thanks for their great work.