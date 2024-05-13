#!/usr/bin/env bash

project_root=/mnt/disks/data/sources/resnet/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet
service_loc=/mnt/disks/data/sources/muyu/easl-utils/tf-data/service/

stop_instance_when_done=false

exp_root="$project_root/results/CACHEW"

#exp_profile_dir="Apr12_Retina_Profile"

# rm -r $exp_root
mkdir -p $exp_root
export PYTHONPATH=/mnt/disks/data/sources/resnet/ml_input_processing/experiments/ml/models

function start_cluster {(
  local workers=${1}

  cd ${service_loc}

  echo "Deploying service with ${workers} workers..."
  sed "s/num_workers:[ \t]\+[0-9]\+/num_workers: $workers/g" "${service_loc}/default_config.yaml" \
     > ${service_loc}/temp_config.yaml
  python3 ${service_loc}/service_deploy.py --config=${service_loc}/temp_config.yaml
  echo "Done deploying service!"

  cd ${project_root}
)}

function restart_service {(
  local policy=${1}
  local workers=${2}
  cd ${service_loc}

  echo "Restarting service with ${workers} workers..."
  sed "s/scaling_policy:[ \t]\+[0-9]\+/scaling_policy: $policy/g" "${service_loc}/default_config.yaml" \
     > ${service_loc}/temp_config_.yaml
  sed "s/num_workers:[ \t]\+[0-9]\+/num_workers: $workers/g" "${service_loc}/temp_config_.yaml" \
   > ${service_loc}/temp_config.yaml
  python3 ${service_loc}/service_deploy.py --config=${service_loc}/temp_config.yaml --restart
  echo "Done restarting service!"

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
  python3 ${service_loc}/service_deploy.py --config=${service_loc}/temp_config.yaml --stop
  echo "Service torn down!"

  cd ${project_root}
)}


start_cluster 8
for policy in 1
do
  restart_service $policy 8
  exp_name="policy-$policy"
  this_exp_dir="$exp_root/$exp_name"
  mkdir -p $this_exp_dir

  local_worker_count=5
  if [ $policy -eq 1 ]; then
      local_worker_count=0
  fi

  cp $0 $this_exp_dir

  dispatcher_name=$( kubectl get nodes | head -n 2 | tail -n 1 | awk '{print $1}' )
  echo "Dispatcher name is $dispatcher_name"

  prep_src_file="$project_root/imagenet_preprocessing.py"
  breakpoint_idx=4
  sed -i "s/DISPATCHER_IP=['\"][a-zA-Z0-9-]*['\"]/DISPATCHER_IP='${dispatcher_name}'/" "${prep_src_file}"
  sed -i "s/^DISTRIBUTE_CHOICE[ \t]\+=[ \t]\+[0-9]\+/DISTRIBUTE_CHOICE = ${breakpoint_idx}/g" "${prep_src_file}"

  dispatcher_pod_name=$( kubectl get pods | head -n 2 | tail -n 1 | awk '{print $1}' )
  echo "Dispatcher pod name is $dispatcher_pod_name"

  bash $service_loc/stat_scripts/stat_all.sh $this_exp_dir

  export EASL_MUYU_FROM_WHICH_WORKER_METRICS="$this_exp_dir/tp.log"

  echo "Start Training..."
  main_program_log_path="$this_exp_dir/main.log"
  bash $project_root/run_imageNet_TPU.sh $exp_name $main_program_log_path $local_worker_count $dispatcher_name

  echo "Ended Training..."

  bash $service_loc/stat_scripts/dispatcher_log.sh $this_exp_dir
  bash $service_loc/stat_scripts/kill_stat_all.sh $this_exp_dir

done
terminate_cluster $this_exp_dir
