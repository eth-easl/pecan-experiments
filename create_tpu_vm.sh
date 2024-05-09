#!/usr/bin/env bash

my_user=${1:-${USER}}
vm_name="atc24-ae-${my_user}"

while ! gcloud alpha compute tpus tpu-vm create --zone=europe-west4-a --accelerator-type="v2-8" --version='tpu-vm-tf-2.8.0' ${vm_name}; do echo "Trying again..."; done

if [ "$?" -ne 0 ]; then
  echo "Successfully created TPU VM ${vm_name}"
else
  echo "Could not create TPU VM ${vm_name}"
fi