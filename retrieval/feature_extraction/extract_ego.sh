srun -p Gvlab-S1 -N1 -n1 --gres=gpu:1 --cpus-per-task=12 --job-name=test \
    python main_feature_extraction.py \
    --model videomae \
    --ego_checkpoint /mnt/petrelfs/xujilan/opensource/Egoinstructor_vinci/retrieval/checkpoints/videomae_L14_checkpoint_best.pth \
    --video_root /mnt/petrelfs/xujilan/opensource/Egoinstructor_vinci/retrieval/assets/videos \
    --output_dir /mnt/petrelfs/xujilan/opensource/Egoinstructor_vinci/retrieval/assets/video_features \
    --video_csv /mnt/petrelfs/xujilan/opensource/Egoinstructor_vinci/retrieval/assets/video_info.csv \
    