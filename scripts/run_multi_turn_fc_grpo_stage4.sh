set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL="${MODEL:-/path/to/your/model}"
TRAIN_FILE="${TRAIN_FILE:-data/bfcl_train.parquet}"
VAL_FILE="${VAL_FILE:-data/bfcl_val.parquet}"

python3 -m verl.trainer.main_ppo \
  --config-path=env_tuning/config \
  --config-name=multi_turn_fc_grpo_stage4 \
  data.train_files=${TRAIN_FILE} \
  data.val_files=${VAL_FILE} \
  actor_rollout_ref.model.path=${MODEL} \
  actor_rollout_ref.ref.path=${MODEL}
