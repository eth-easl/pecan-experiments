# ml_input_processing

# Documentation for the AutoOrder Policy

The source code for the AutoOrder policy is located in the `oto-autoorder` [branch](https://github.com/eth-easl/cachew/tree/oto-autoorder) of the cachew repository.

Compile with Bazel 4.2.1

The code is to be compiled on TPU VM (note that it MUST BE COMPILED SEPARATELY for CPU and TPU). Consult the scripts and instructions [here](https://gitlab.inf.ethz.ch/OU-KLIMOVIC/easl/easl-utils/-/tree/oto-config/tf-data/build) for compilation and [here](https://gitlab.inf.ethz.ch/OU-KLIMOVIC/easl/easl-utils/-/tree/oto-config/tf-data/service/docker) for Docker image generation.

### API

The AutoOrder policy API is very simple, the user does not have to add anything to the pipeline. If you want a certain `map()` op to remain in its place, set the `keep_position` flag to True.

### Implementation documentation details

The majority of the policy is implemented in the Python layer in the [dataset_ops.py file](https://github.com/eth-easl/cachew/blob/oto-autoorder/tensorflow/python/data/ops/dataset_ops.py). The most important logic (determining which op to move where, and the mechanism for op swapping intself) is contained in the ```map()``` method. The [dataset_ops_utils.py file](https://github.com/eth-easl/cachew/blob/oto-autoorder/tensorflow/python/data/ops/dataset_ops_utils.py) contains some helper methods. 

On the C++ side there is the inflation factor feedback loop implemented. They are added to the ClientHeartBeatResponse in the ```ClientHeartbeat()``` method of the ```tensorflow/core/data/service/dispatcher_impl.cc``` file. The inflation factors themselves are calculated in the ```tensorflow/core/data/service/easl/ordering_utils.cc``` file. In addition also the MetadataStore is expanded to store data about the inflation factors.

# Experiments

## Setup

The experiments have been performed on a `v2-8 TPU VM` with `n2-standard-8` workers. Note you may want to change to your NETHZ

Deploy a VM:

```
gcloud alpha compute tpus tpu-vm create otmrazVM \
--zone=europe-west4-a \
--accelerator-type=v2-8 \
--project=tfdata-service \
--version=tpu-vm-tf-2.8.0
```

SSH into it:

```
gcloud alpha compute tpus tpu-vm ssh otmrazVM --zone europe-west4-a
```

Install necessary libraries:

```
curl -LO "https://dl.k8s.io/release/v1.19.7/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client  # should now be 1.19.7

curl -Lo kops https://github.com/kubernetes/kops/releases/download/v1.19.3/kops-linux-amd64
chmod +x ./kops
sudo mv ./kops /usr/local/bin/
kops version # should now be 1.19.3

# Install Glusterfs on VM
sudo add-apt-repository ppa:gluster/glusterfs-9
sudo apt update
sudo apt install -y glusterfs-client
# Mount Glusterfs
gluster_fs_mount=/mnt/disks/gluster_data
sudo mkdir -p $gluster_fs_mount
```

Clone the EASL repos:

```
# Clone easl-utils
git clone --single-branch --branch oto-config https://gitlab.inf.ethz.ch/OU-KLIMOVIC/easl/easl-utils.git
# Authentication

# Clone ml_input_processing
git clone --single-branch --branch otmraz-exp https://gitlab.inf.ethz.ch/OU-KLIMOVIC/easl/ml_input_processing.git
# Authentication
```

Install packages for python: (Please run one by one, some require mannual confirmations)

```
sudo apt update
sudo apt install python3.8 -y
sudo apt-get install python3.8-dev python3.8-venv -y
sudo apt install -y jq
sudo apt install cpulimit
sudo apt-get install ffmpeg libsm6 libxext6 -y
sudo apt-get install -y python-tk
```

Create your python environment:

```
python3.8 -m venv env
source env/bin/activate
cd ~/easl-utils/tf-data/service
pip install --upgrade pip
pip install -r requirements.txt
```

Chose and download the tensorflow wheel you want (most times you want the oto-pecan wheel) and install it:
```
gsutil cp gs://otmraz-eu-logs/CompiledTF/oto-pecan/TPU/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl .
```

```
python -m pip install --force-reinstall tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl

# Downgrade protobuf
pip install protobuf==3.20.0 werkzeug==2.1.1 markupsafe==2.0.1 googleapis-common-protos==1.56.4
pip install imgaug tensorflow_model_optimization tensorflow_addons tensorflow_datasets pandas
```

As the basis of our setup we use the `otmraz-exp` branch of the [Cachew artifact-eval scripts](https://github.com/eth-easl/cachew_experiments/tree/otmraz-exp). Consult the READMEs in that repo for further information & troubleshooting.

Clone the repo and go into the `experiments/autocaching` directory. The [default_config.yaml](https://github.com/eth-easl/cachew_experiments/blob/otmraz-exp/experiments/autocaching/default_config.yaml) allows you to specify parameters of the Cachew service. Most importantly, the image (`tf_oto:ao` for AutoOrder, `tf_oto:lw` for the AutoPlacement policy) and the scaling policy (3 for Cachew AutoScaling, 2 for fixed no. of workers).

```
cd ~
git clone --single-branch --branch otmraz-exp https://github.com/eth-easl/cachew_experiments.git && cd cachew_experiments
```

Start the kubernetes cluster:
```
cd ~/cachew_experiments/experiments/autocaching
./manage_cluster.sh start
```

Get the name of the dispatcher node using the following command and use it inside the model scripts (None: As of 23.5.23 this should be done by the scripts automatically).
```
kubectl get node
```

## Throughput / cost experiments

### ResNet

Inside ```resnet_ctl_imagenet_main.py``` and ```imagenet_preprocessing.py``` set the correct dipatcher name and choose the desired no. of local workers.

Run the following commands to set up:
```
cd ~/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet

preprocessing_file="imagenet_preprocessing.py"
logging_file="exp_log"+time.strftime("%Y_%m_%d-%H_%M_%S")+".txt"

export PYTHONPATH=$HOME/ml_input_processing/experiments/ml/models/
export TF_DUMP_GRAPH_PREFIX=$HOME/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet/graph_dump.log

tpu_name="local"
model_dir="gs://otmraz-eu-logs/Resnet/ImageNet/FINAL"
data_dir="gs://tfdata-imagenet-eu" # This scripts needs 2 subfolders: train, validation

train_epochs=90
log_out="resnet_test.log"
```
**Depending on the experiment you are running set:**
* the number of local wokrers to be spawned (0 or 10)
* the flag whether or not to use AutoOrder (True or False)
* the DISPATCHER_IP ('disp' for using a Cachew/Pecan service, the script should find the dispatcher name automatically. Or "None" for preprocessing without a service (e.g. for Fig. 1))
```
export USE_AUTOORDER=True
export n_loc=10
export DISPATCHER_IP='disp'
```


The following commad performs the experiment:
```
python3 resnet_ctl_imagenet_main.py --enable_checkpoint_and_export=true --tpu=$tpu_name --model_dir=$model_dir --data_dir=$data_dir --batch_size=1024 --steps_per_loop=500 --train_epochs=$train_epochs --use_synthetic_data=false --dtype=fp32 --enable_eager=true --enable_tensorboard=true --distribution_strategy=tpu --log_steps=50 --single_l2_loss_op=true --verbosity=0 --skip_eval=true --use_tf_function=true --num_local_workers=$n_loc 2>&1 | tee $log_out
```

### SimCLR

Inside ```data.py``` and ```run.py``` set the correct dipatcher name and choose the desired no. of local workers.

Run the following commands to set up:
```
cd ~/ml_input_processing/experiments/ml/simclr

dataset="imagenet"
stop_instance_when_done=false

export USE_TPU=true
export TPU_NAME="local"
export TPU_VM=true
export STORAGE_BUCKET="gs://tfdata-datasets-eu/imagenet_tfds/imagenet_tfds"
export DATA_DIR="$STORAGE_BUCKET/"
export MODEL_DIR="gs://otmraz-eu-logs/Resnet/simCLR/FINAL"
CACHE_DIR=$HOME/training-data/cache_temp

caching=false #true
caching_period=1
blur_in_model=false
augment=true
shuffle=false #true
exp_name="cache-1w"
batch_size=512 #1024
epochs=100
log_out="simclr_test.log"
dispatcher_ip=None

export PYTHONPATH=$HOME/ml_input_processing/experiments/ml/models
```

**Depending on the experiment you are running set:**
* the number of local wokrers to be spawned (0 or 10)
* the flag whether or not to use AutoOrder (True or False)
* the DISPATCHER_IP ('disp' for using a Cachew/Pecan service, the script should find the dispatcher name automatically Or "None" for preprocessing without a service (e.g. for Fig. 1))
```
export USE_AUTOORDER=True
n_loc=10
export DISPATCHER_IP='disp'
```

The following commad performs the experiment:
```
python3 run.py --mode=train --train_mode=pretrain \
      --train_batch_size=$batch_size --train_epochs=$epochs --temperature=0.1 \
      --learning_rate=0.075 --learning_rate_scaling=sqrt --weight_decay=1e-4 \
      --dataset=imagenet2012 --image_size=224 --eval_split=validation \
      --proj_head_mode=linear --data_dir=$DATA_DIR --model_dir=$MODEL_DIR \
      --use_tpu=$USE_TPU --tpu_name=$TPU_NAME --tpu_vm=$TPU_VM \
      --caching=$caching --caching_period=$caching_period --cache_dir=$CACHE_DIR \
      --blur_in_model=$blur_in_model --augment=$augment --shuffle_files=$shuffle \
      --checkpoint_steps 2502 --num_local_workers=$n_loc 2>&1 | tee $log_out
```

### RetinaNet

Inside ```main.py``` and ```dataloader/input_reader.py``` set the correct dipatcher name and choose the desired no. of local workers.

Run the following commands to set up:
```
cd ~/ml_input_processing/experiments/ml/models/official/vision/detection

pip install -r ~/ml_input_processing/experiments/ml/models/official/requirements.txt

TPU_NAME="local"
TPU_VM=true
TPU_ADDRESS="local"
DATA_DIR="gs://tfdata-datasets-eu/coco"
TRAIN_FILE_PATTERN="$DATA_DIR/train-*"
EVAL_FILE_PATTERN="$DATA_DIR/val-*"
VAL_JSON_FILE="gs://tfdata-datasets-eu/coco/raw-data/annotations/instances_val2017.json"
RESNET_CHECKPOINT="gs://cloud-tpu-checkpoints/retinanet/resnet50-checkpoint-2018-02-07"

ITERS_PER_LOOP=1848 # 1848 steps = 1 epoch

epochs=72 # default: 13, 72 is 2nd version
total_steps=$(($epochs * $ITERS_PER_LOOP))
eval=true

save_checkpoint_freq=1

model_dir="gs://otmraz-eu-logs/Retinanet/FINAL"
log_out="main.log"
export PYTHONPATH=$HOME/ml_input_processing/experiments/ml/models/
```

**Depending on the experiment you are running set:**
* the number of local wokrers to be spawned (0 or 10)
* the flag whether or not to use AutoOrder (True or False)
* the DISPATCHER_IP ('disp' for using a Cachew/Pecan service, the script should find the dispatcher name automatically Or "None" for preprocessing without a service (e.g. for Fig. 1))
```
export USE_AUTOORDER=True
n_loc=10
export DISPATCHER_IP='disp'
```

The following commad performs the experiment:
```
python main.py --strategy_type=tpu --tpu="${TPU_ADDRESS?}" --model_dir="${model_dir?}" --save_checkpoint_freq=$save_checkpoint_freq --mode=train --local_workers=$n_loc --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?}, iterations_per_loop: ${ITERS_PER_LOOP}, total_steps: ${total_steps}}, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?}, num_steps_per_eval: ${ITERS_PER_LOOP} } }" 2>&1 | tee $log_out
```

## Accuracy experiments

Due to the large runtimes till convergenece, the accuracy experiments are run without a Cachew service, and the code is in the otmraz-resnet [branch](https://gitlab.inf.ethz.ch/OU-KLIMOVIC/easl/ml_input_processing/-/tree/otmraz-resnet/)

### Visualizations

The `paper_plots` [directory](https://gitlab.inf.ethz.ch/OU-KLIMOVIC/easl/ml_input_processing/-/tree/otmraz-exp/paper_plots) contains various scripts for plotting experiment results. It is advisable to run these with the VS Code debugger, as the figure dimensions, labels, axes, etc. should be adjusted to the data.

### CPU / MEM measuring

[This script](https://gitlab.inf.ethz.ch/OU-KLIMOVIC/easl/ml_input_processing/-/blob/otmraz-exp/experiments/ml/models/official/vision/image_classification/resnet/cpu_mem.sh) measures the CPU and Memory utilization at a milisecond granularity.

### Motivating experiments

The motivating experiments were performed solely with Muyu Li's [implemntation](https://github.com/eth-easl/cachew/tree/muyu-thesis-local-worker) of the AutoPlacement policy and the pipelines for the experiments were reordered manually. See [this](https://drive.google.com/drive/folders/1FcMDSCcDpKblpywMnouXFCrephany9rZ?usp=share_link) Google Drive folder which contains the original and reordered input pipelines and Muyu Li's [README](https://gitlab.inf.ethz.ch/OU-KLIMOVIC/easl/ml_input_processing/-/blob/otmraz-exp/Muyu-thesis-README.md) for further details on the experiment setup.

### Cross-Zone training

For the cross-zone training we need to Google Kubernetes Engines (in order for the dispatcher and workers to be able to communicate across different zone).

Use [these](https://github.com/tensorflow/ecosystem/blob/master/data_service/README.md) instructions and scripts for setup.

Note that in our experiments we only used 2 (close or remote workers). The data was stored in a ```us-central-1``` bucket and the TPU was located in the ```europe-west4-a``` zone.
