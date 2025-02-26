SRC_DOMAIN=$1
TGT_DOMAIN=$2
SRC_MODEL_DIR=$3

PORT=10000
MEMO="target"
SEED="2022"

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python main_adacontrast.py \
seed=${SEED} port=${PORT} memo=${MEMO} project="domainnet-126" \
learn.num_neighbors=5 \
learn.mix_prob=0.8 \
learn.patch_height=112 \
learn.alpha_spm=8 \
data.batch_size=128 \
data.aug_type="shuffle_patch_mix" \
data.data_root="${PWD}/datasets" data.workers=8 \
data.dataset="domainnet-126" data.source_domains="[${SRC_DOMAIN}]" data.target_domains="[${TGT_DOMAIN}]" \
model_src.arch="resnet50" \
model_tta.src_log_dir=${SRC_MODEL_DIR} \
optim.lr=2e-4
