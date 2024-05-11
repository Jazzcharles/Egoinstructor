srun -p video3 -N1 -n1 --gres=gpu:1 --cpus-per-task=14 --quotatype=auto --job-name=our -x SH-IDC1-10-140-37-104 \
python extract_feature.py \
    --config configs/retriever/extract_text_feat.yml \