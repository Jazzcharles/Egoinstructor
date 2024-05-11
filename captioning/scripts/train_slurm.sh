PRETRAINED_MODEL_NAME_OR_PATH=luodian/OTTER-MPT1B-RPJama-Init
DATAPATH_EGO=/path/to/your/egovideos/
DATAPATH_EXO=/path/to/your/exovideos/
METAPATH=/path/to/your/metadata.json
TRAIN_CONFIG_PATH=/path/to/your/train_config.json
MAX_SHOT=4
BATCH_SIZE=4
GPUS=8

export PYTHONPATH=.
srun -p Gvlab-S1 -N1 -n1 --gres=gpu:${GPUS} --cpus-per-task=12 --quotatype=auto \
accelerate launch --config_file=./configs/accelerate_config_ddp_${GPUS}gpu.yaml \
    main.py \
    --pretrained_model_name_or_path=${PRETRAINED_MODEL_NAME_OR_PATH} \
    --datapath_ego=${DATAPATH_EGO} \
    --datapath_exo=${DATAPATH_EXO} \
    --metapath=${METAPATH} \
    --train_config_path=${TRAIN_CONFIG_PATH} \
    --batch_size=${BATCH_SIZE} \
    --num_epochs=5 \
    --report_to_wandb \
    --wandb_entity=test \
    --run_name=OTTER-MPT1B-RPJama-4SHOT-8GPU-BS32 \
    --wandb_project=OTTER-MPT1B \
    --workers=16 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --offline \
    --xview \
    --max_shot=${MAX_SHOT} \
    --save_ckpt_each_epoch \