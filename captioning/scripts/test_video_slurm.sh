# PRETRAINED_NAME_OR_PATH=luodian/OTTER-MPT1B-RPJama-Init
PRETRAINED_NAME_OR_PATH=/mnt/petrelfs/xujilan/.cache/huggingface/hub/models--luodian--OTTER-MPT1B-RPJama-Init/snapshots/74490d8a17c2db46290a19b33229c7a2c62a8528/
PRETRAINED_CKPT=/mnt/petrelfs/xujilan/checkpoints/Otter/OTTER-MPT1B-RPJama-xview-4shot-8gpu-debug-1ep/egoinstructor_captioner_4shot.pt
# PRETRAINED_CKPT=/mnt/petrelfs/xujilan/tools/Otter/checkpoints/OTTER-MPT1B-RPJama-xview-4shot-new-8gpu/checkpoint_0.pt
MAX_SHOT=4

export PYTHONPATH=.
srun -p Gvlab-S1 -N1 -n1 --gres=gpu:1 --cpus-per-task=12 --quotatype=auto \
    python test_video.py \
    --testdata ./assets/testdata.json \
    --pretrained_name_or_path=${PRETRAINED_NAME_OR_PATH} \
    --pretrained_checkpoint=${PRETRAINED_CKPT} \
    --max_shot=${MAX_SHOT} \