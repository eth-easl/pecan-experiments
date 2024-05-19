#!/bin/bash
source ~/atc_venv/bin/activate
cd ~/pecan-experiments/experiments/pecan

echo 1. Running Pecan

new_line='image: "tf_oto:dan_fast_removal_100b"'
sed -i.bak -E 's#^image:.*#'"$new_line"'#' default_config.yaml
./manage_cluster.sh restart_service

export USE_AUTOORDER=True
export n_loc=10
export DISPATCHER_IP='disp'
num_epochs=7
log_out="../../../../../../../../pecan-experiments/experiments/pecan/logs/retina_pecan.log"

TPU_NAME="local"
TPU_VM=true
TPU_ADDRESS="local"
DATA_DIR="gs://tfdata-datasets-eu/coco"
TRAIN_FILE_PATTERN="$DATA_DIR/train-*"
EVAL_FILE_PATTERN="$DATA_DIR/val-*"
VAL_JSON_FILE="gs://tfdata-datasets-eu/coco/raw-data/annotations/instances_val2017.json"
RESNET_CHECKPOINT="gs://cloud-tpu-checkpoints/retinanet/resnet50-checkpoint-2018-02-07"
ITERS_PER_LOOP=1848 # 1848 steps = 1 epoch
epochs=$num_epochs # default: 13, 72 is 2nd version
total_steps=$(($epochs * $ITERS_PER_LOOP))
eval=false
save_checkpoint_freq=1
model_dir="gs://otmraz-eu-logs/Retinanet/${USER}"
export PYTHONPATH=$HOME/pecan-experiments/ml_input_processing/experiments/ml/models/
total_steps=$(($epochs * $ITERS_PER_LOOP))

# Clean out model directory
gsutil rm -r $model_dir

# Model training itself
cd ../../ml_input_processing/experiments/ml/models/official/vision/detection/
#python main.py --strategy_type=tpu --tpu="${TPU_ADDRESS?}" --model_dir="${model_dir?}" --save_checkpoint_freq=$save_checkpoint_freq --mode=train --local_workers=$n_loc --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?}, iterations_per_loop: 250, total_steps: ${total_steps}}, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?}, num_steps_per_eval: ${ITERS_PER_LOOP} } }" 2>&1 | tee $log_out

#cd ../../../../../../../../pecan-experiments/experiments/pecan
