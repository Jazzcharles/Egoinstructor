PRETRAINED_MODEL_NAME_OR_PATH=luodian/OTTER-MPT1B-RPJama-Init
DATAPATH_EGO=/path/to/your/egovideos/
DATAPATH_EXO=/path/to/your/exovideos/
METAPATH=/path/to/your/metadata.json
TRAIN_CONFIG_PATH=/path/to/your/train_config.json
TRAINED_CKPT=/path/to/your/trained_checkpoint.pt
MAX_SHOT=4
BATCH_SIZE=1
GPUS=1

export PYTHONPATH=.
srun -p Gvlab-S1 -N1 -n1 --gres=gpu:1 --cpus-per-task=12 --quotatype=auto \
accelerate launch --config_file=./configs/accelerate_config_ddp_1gpu.yaml \
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
    --run_name=OTTER-MPT1B-RPJama-test \
    --wandb_project=OTTER-MPT1B \
    --workers=16 \
    --lr_scheduler=cosine \
    --learning_rate=1e-5 \
    --warmup_steps_ratio=0.01 \
    --offline \
    --testonly \
    --trained_ckpt=${TRAINED_CKPT} \
    --xview \
    --max_shot=${MAX_SHOT} \
