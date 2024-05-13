#!/bin/bash

# 1: exp_name
# 2: log_dir
# 3: n_local_workers
# 4: dispatcher_ip

timestamp=$(date "+%Y-%m-%d-%Hh%Mm%S")
exp_name="$1"
model_dir="gs://tfdata-train-logs/resnet50/ImageNet/$timestamp-$exp_name"

data_dir="gs://tfdata-imagenet" # ImageNet from GCS


epochs=1
target_accuracy=2 #"0.749"
stop_instance_when_done=false ##################WARNING##################

num_training_images=1281167 # ImageNet


num_gpus=4
per_gpu_batch_size=312
batch_size=$(($per_gpu_batch_size * $num_gpus))
steps_per_loop=$(($num_training_images / $batch_size + 1))
enable_checkpoint_and_export="false" # save model
skip_eval=true

# remove old snapshots
rm -r /training-data/cache_temp/snapshot*

# commandline flags to the training script
params=""
params+="--num_gpus=$num_gpus "
params+="--dtype=fp16 "
params+="--batch_size=$batch_size "
params+="--train_epochs=$epochs "
params+="--target_accuracy=$target_accuracy "
params+="--skip_eval=$skip_eval "

params+="--epochs_between_evals=1 "
#params+="--optimizer=SGD "
#params+="--lr_schedule=polynomial "
#params+="--label_smoothing=0.1 "
#params+="--weight_decay=0.0002 "

params+="--report_accuracy_metrics=true "
params+="--log_steps=125 "
params+="--enable_tensorboard=false "

params+="--steps_per_loop=$steps_per_loop "
params+="--enable_eager=true "
params+="--tf_gpu_thread_mode=gpu_private " #gpu_private / gpu_shared / global
#params+="--per_gpu_thread_count=1 "
params+="--datasets_num_private_threads=32 "
params+="--data_dir=$data_dir "
params+="--model_dir=$model_dir "
params+="--enable_checkpoint_and_export=$enable_checkpoint_and_export "
params+="--single_l2_loss_op "
params+="--verbosity=1 "
#params+="--nouse_tf_function --nouse_tf_while_loop"

params+="--num_local_workers=$3 "
params+="--dispatcher_ip=$4 "


# Launch training
cmd="python3 resnet_ctl_imagenet_main.py ${params[@]}"
log_out="$2"

echo "Running experiment '$exp_name'"
echo "Training model and piping output to $log_out ..."
${cmd[@]} 2>&1 | tee $log_out

echo ""
echo "Finished training for experiment:$exp_name!"



