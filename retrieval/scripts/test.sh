srun -p Gvlab-S1 -N1 -n1 --gres=gpu:1 --cpus-per-task=14 --quotatype=reserved --job-name=test \
python main_pretrain_contrastive.py \
    --config ./configs/test.yml \