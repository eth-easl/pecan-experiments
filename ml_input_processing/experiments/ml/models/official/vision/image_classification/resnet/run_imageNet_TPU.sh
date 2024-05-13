#!/usr/bin/env bash

timestamp=$(date "+%Y-%m-%d-%Hh%Mm%S")
exp_name=$1
model_dir="gs://tfdata-train-logs/resnet50/ImageNet/$timestamp-$exp_name"

# Choose the dataset directory here
#data_dir="/training-data/imagenet2012/tfrecords" # ImageNet from SSD
data_dir="gs://tfdata-imagenet" # ImageNet from GCS
#data_dir="/training-data/imagenet-tiny/tfrecords" # Tiny
#data_dir="/training-data/imagenet_test/tfrecords" # Test dataset with 4 images

num_training_images=1281167 # ImageNet
#num_training_images=100000 # Tiny ImageNet
#num_training_images=4 # Test dataset

epochs=4
batch_size=1024
#steps_per_loop=500
steps_per_loop=$(($num_training_images / $batch_size + 1))
tpu_name="local"  # "dan-inst-tpu"
#tpu_name="muyu-tpu-vm"
enable_checkpoint_and_export="false" # save model
skip_eval="true"

# remove old snapshots
rm -r /training-data/cache_temp/snapshot*

# commandline flags to the training script
params=""
params+="--dtype=fp32 "
params+="--tpu=$tpu_name "
params+="--batch_size=$batch_size "
params+="--train_epochs=$epochs "
params+="--distribution_strategy=tpu "
params+="--use_synthetic_data=false "
params+="--target_accuracy=2 "
params+="--skip_eval=$skip_eval "


params+="--report_accuracy_metrics=true "
params+="--log_steps=125 "
params+="--enable_tensorboard=false "

params+="--steps_per_loop=$steps_per_loop "
params+="--enable_eager=true "
#params+="--datasets_num_private_threads=32 "
params+="--data_dir=$data_dir "
params+="--model_dir=$model_dir "
params+="--enable_checkpoint_and_export=$enable_checkpoint_and_export "
params+="--single_l2_loss_op=true "
params+="--verbosity=0 "

params+="--num_local_workers=$3 "
params+="--dispatcher_ip=$4 "

# Launch training
cmd="python3 resnet_ctl_imagenet_main.py ${params[@]}"
log_out=$2

echo "Running experiment '$exp_name'"
echo "Training model and piping output to $log_out ..."
${cmd[@]} 2>&1 | tee $log_out

echo ""
echo "Finished training!"

