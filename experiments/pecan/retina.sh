#!/bin/bash

# Source environment variables
source ~/.bashrc

# Predefined Variables
TPU_NAME=${TPU_NAME:-"local"}
TPU_VM=${TPU_VM:-true}
TPU_ADDRESS=${TPU_ADDRESS:-"local"}
DATA_DIR=${DATA_DIR:-"gs://tfdata-datasets-eu/coco"}
TRAIN_FILE_PATTERN="${DATA_DIR}/train-*"
EVAL_FILE_PATTERN="${DATA_DIR}/val-*"
VAL_JSON_FILE=${VAL_JSON_FILE:-"gs://tfdata-datasets-eu/coco/raw-data/annotations/instances_val2017.json"}
RESNET_CHECKPOINT=${RESNET_CHECKPOINT:-"gs://cloud-tpu-checkpoints/retinanet/resnet50-checkpoint-2018-02-07"}
ITERS_PER_LOOP=${ITERS_PER_LOOP:-1848} # 1848 steps = 1 epoch

# Command-line Arguments
epochs=${1:-13} # default: 13
eval=${2:-false}
save_checkpoint_freq=${3:-1}
model_dir=${4:-"gs://otmraz-eu-logs/Retinanet/${USER}"}
log_out=${5:-"./train.log"}

total_steps=$(($epochs * $ITERS_PER_LOOP))

export PYTHONPATH=$HOME/pecan-experiments/ml_input_processing/experiments/ml/models/

# Model training itself
cd ../../ml_input_processing/experiments/ml/models/official/vision/detection/
python main.py --strategy_type=tpu --tpu="${TPU_ADDRESS?}" --model_dir="${model_dir?}" --save_checkpoint_freq=$save_checkpoint_freq --mode=train --local_workers=$n_loc --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?}, iterations_per_loop: 250, total_steps: ${total_steps}}, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?}, num_steps_per_eval: ${ITERS_PER_LOOP} } }" 2>&1 | tee $log_out
