#!/usr/bin/env bash

project_root=/mnt/disks/data/sources/muyu/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet
service_loc=/mnt/disks/data/sources/muyu/easl-utils/tf-data/service/

stop_instance_when_done=false

exp_root="$project_root/results/profiling"

rm -r $exp_root
mkdir -p $exp_root
export PYTHONPATH=/mnt/disks/data/sources/muyu/ml_input_processing/experiments/ml/models

function start_cluster {(
  local workers=${1}

  cd ${service_loc}

  echo "Deploying service with ${workers} workers..."
  sed "s/num_workers:[ \t]\+[0-9]\+/num_workers: $workers/g" "${service_loc}/default_config.yaml" \
     > ${service_loc}/temp_config.yaml
  python ${service_loc}/service_deploy.py --config=${service_loc}/temp_config.yaml
  echo "Done deploying service!"

  cd ${project_root}
)}

function terminate_cluster {(
  local this_exp_dir=${1}

  cd ${service_loc}

  # Log the node activity
  cluster_log_dir=${this_exp_dir}/cluster
  mkdir -p ${cluster_log_dir}
  nodes="$( kubectl get nodes )"
  echo "${nodes}" > ${cluster_log_dir}/nodes.txt

  echo "${nodes}" | tail -n +2 | while read line; do
    node_name=$( echo ${line} | awk '{print $1}' )
    kubectl describe nodes ${node_name} > ${cluster_log_dir}/${node_name}_describe.txt
  done


  dispatcher_pod_name=$( kubectl get pods | head -n 2 | tail -n 1 | awk '{print $1}' )
  kubectl logs ${dispatcher_pod_name} > ${cluster_log_dir}/dispatcher.log

  sleep 50

  # Stop the service
  echo "Tearing down service..."
  python ${service_loc}/service_deploy.py --config=${service_loc}/temp_config.yaml --stop
  echo "Service torn down!"

  cd ${project_root}
)}

for remote_worker_count in 4 5
do
  for local_worker_count in 0 1 2
do
  start_cluster $remote_worker_count

#remote_worker_count=2
#local_worker_count=2

  exp_name="rwc-$remote_worker_count-lwc-$local_worker_count"
  this_exp_dir="$exp_root/$exp_name"
  mkdir -p $this_exp_dir

  dispatcher_name=$( kubectl get nodes | head -n 2 | tail -n 1 | awk '{print $1}' )
  echo "Dispatcher name is $dispatcher_name"

  prep_src_file="$project_root/imagenet_preprocessing.py"
  sed -i "s/DISPATCHER_IP=['\"][a-zA-Z0-9-]*['\"]/DISPATCHER_IP='${dispatcher_name}'/" "${prep_src_file}"

  dispatcher_pod_name=$( kubectl get pods | head -n 2 | tail -n 1 | awk '{print $1}' )
  echo "Dispatcher pod name is $dispatcher_pod_name"

  echo "Start Logging Dispatcher"
  dispatcher_log_dir="$this_exp_dir/cluster/dispatcher"
  mkdir -p $dispatcher_log_dir
  bash $project_root/dispatcher_log.sh $dispatcher_log_dir $dispatcher_pod_name & # run dispatcher dump periodically

  echo "Start Recording Cpu Stat"
  cpu_stat_path="$this_exp_dir/cpu_stat.txt"
  bash $project_root/cpu_stat.sh $cpu_stat_path & # dump cpu util rate

  echo "Start Recording Network Bandwidth"
  nw_stat_path="$this_exp_dir/nw_stat.txt"
  sudo iftop -t > $nw_stat_path & # dump network usage

  remote_workers=`kubectl logs $dispatcher_pod_name | sed -n 's/.*Registering new worker at address \(.*\)$/\1/p'`

  profile_log_path="gs://muyu-kops-store/Mar30/$exp_name"
  profile_path="$this_exp_dir/profile.log"
  echo "Start Profiling"
  python capture_profile.py $profile_log_path $local_worker_count $remote_workers > $profile_path &

  echo "Start Training..."
  main_program_log_path="$this_exp_dir/main.log"

  echo bash run_imageNet_sweep.sh $exp_name $main_program_log_path $local_worker_count $dispatcher_name
  bash run_imageNet_sweep.sh $exp_name $main_program_log_path $local_worker_count $dispatcher_name

  echo "Ended Training..."

  echo "Kill Logging Dispatcher"
  pkill -f "bash $project_root/dispatcher_log.sh $dispatcher_log_dir $dispatcher_pod_name"

  echo "Kill Recording Cpu Stat"
  pkill -f "bash $project_root/cpu_stat.sh $cpu_stat_path"

  echo "Kill Recording NW Stat"
  sudo pkill -f "iftop"

  echo "Killing Cluster for $exp_name"
  terminate_cluster $this_exp_dir
done
done

if [ "$stop_instance_when_done" = true ] ; then
    echo "WARNING: Deleting training disk and stopping instance in 60 seconds..."
    for i in $(seq 1 1 59); do
        sleep 1
        echo $((60-$i))
    done

    sudo umount /dev/sdb /training-data

    DISK_NAME="$HOSTNAME-training-data"
    echo "Detaching $DISK_NAME disk..."
    gcloud compute instances detach-disk $HOSTNAME --disk $DISK_NAME --zone us-central1-a

    echo "Deleting $DISK_NAME disk..."
    gcloud compute disks delete $DISK_NAME --zone us-central1-a --quiet

    # stop instance
    gcloud compute instances stop $HOSTNAME --zone us-central1-a
fi

