# Preparing data

## Pre-training data
### Ego4d video features
- Download and extract Ego4d videomae_L14 video features from 🤗[huggingface dataset](https://huggingface.co/datasets/Jazzcharles/ego4d_videomae_L14_feature_fps8). 

### Ego4d metadata
- The metadata for paired ego4d-howto100m is available at 🤗[huggingface dataset](https://huggingface.co/datasets/Jazzcharles/ego4d_train_pair_howto100m). 

```
git lfs install
git clone https://huggingface.co/datasets/Jazzcharles/ego4d_train_pair_howto100m
cd ego4d_train_pair_howto100m/
cp ego4d_train_pair_howto100m.json ./metadata/
```

### HowTo100M video features 
- Download video features following [Temporal Alignment Network](https://github.com/TengdaHan/TemporalAlignNet/blob/main/htm_zoo/visual/download_internvideo_script.sh). 

### HowTo100M metadata
- The metadata containing the caption refined by LLM is available at 🤗 [huggingface dataset](https://huggingface.co/datasets/Jazzcharles/HowTo100M_llama3_refined_caption).
```
git lfs install
git clone https://huggingface.co/datasets/Jazzcharles/HowTo100M_llama3_refined_caption
cd HowTo100M_llama3_refined_caption/
cp htm_llama3_refined.json ./metadata/
```

## Downstream evaluation data

### Video features

| Video features | Link | Size |
|-------------------------|--------|--------|
| epic_kitchen_videomae_L14_feature_fps8 | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/epic_kitchen_videomae_L14_feature_fps8/blob/main/epic_kitchen_videomae_L14_feature_fps8.tar.gz) | 509M
| youcook2_internvideo_MM_L14_features_fps8 | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/youcook2_internvideo_MM_L14_features_fps8/blob/main/youcook2_internvideo_MM_L14_features_fps8.tar.gz) | 385M
| charadesego_videomae_L14_feature_fps8 | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/charadesego_videomae_L14_feature_fps8/blob/main/charadesego_videomae_L14_feature_fps8.tar.gz) | 325M
| charadesego_internvideo_MM_L14_features_fps8 | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/charadesego_internvideo_MM_L14_features_fps8/blob/main/charadesego_internvideo_MM_L14_features_fps8.tar.gz) | 651M
| egolearn_videomae_L14_feature_fps8 | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/egolearn_videomae_internvideo_features/blob/main/egolearn_videomae_L14_feature_fps8.tar.gz) | 368M
| egolearn_exo_internvideo_MM_L14_features | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/egolearn_videomae_internvideo_features/blob/main/egolearn_exo_internvideo_MM_L14_features.tar.gz) | 174M


**Step1**. You can download the video features manually from the given link or use the following command 
```
git lfs install
git clone https://huggingface.co/datasets/Jazzcharles/epic_kitchen_videomae_L14_feature_fps8
cd epic_kitchen_videomae_L14_feature_fps8/
tar -zxvf epic_kitchen_videomae_L14_feature_fps8.tar.gz
```
- For epic-kitchen dataset, download the relevancy matrix from [LaViLA](https://dl.fbaipublicfiles.com/lavila/metadata/EK100/caption_relevancy_EPIC_100_retrieval_test.pkl) and put it under ./metadata.

- For egolearn dataset, please refer to [EgoExoLearn](https://github.com/OpenGVLab/EgoExoLearn) for the original videos.

<br><br>
**Step2**. Modify the feature path in the config files (e.g. configs/default_feature.yml)
```
- ek100_mir:
    - root: /path/to/your/ek100_feature/
```

### Metadata 
| Metadata | Link | Size |
|-------------------------|--------|--------|
| EPIC_100_retrieval_test.csv | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/EPIC_100_retrieval_test.csv) | 1.2M
| EPIC_100_retrieval_test_sentence.csv | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/EPIC_100_retrieval_test_sentence.csv) | 110K
| charades_video_retrieval.json | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/charades_video_retrieval.json) | 30K
| youcook_validation_clip.json | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/youcook_validation_clip.json) | 736K
| youcook_validation_video.json | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/youcook_validation_video.json) | 228K
| egolearn_mcq_validation.json) | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/egolearn_mcq_validation.json) | 1.3M
| ego4d_val_summarization_mcq.json) | 🤗 [HF link](https://huggingface.co/datasets/Jazzcharles/Egoinstructor_downstream_metadata/blob/main/ego4d_val_summarization_mcq.json) | 3.9M

- Download all the metafiles and put them under ./metadata/

