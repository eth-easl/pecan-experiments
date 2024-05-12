import os, sys
import subprocess as sp
from subprocess import Popen, PIPE
import shlex

'''
    a. **Running the input pipeline with Pecan** - producing data for the brown bars
    b. **Runnign the input pipeline with Cachew** - producing data for the orange bars
    c. **Runing the pipeline in the collocated mode** - producing data for the blue bars
    d. **Generating a plot with the cost of each config**. generating a plot `fig8_ResNet50_v2-8.pdf`
'''

restart_workers_cmd = '''./manage_cluster.sh restart_service'''
stop_workers_cmd = '''./manage_cluster.sh stop'''
service_yaml = '''default_config.yaml'''

pecan_img = '''tf_oto:dan_fast_removal_100b''' #'''tf_oto:pecan'''
cachew_img = '''tf_oto:pecan''' # We use the pecan img, but we disable autoorder and simply spawn 0 remote workers

resnet_model_dir = "gs://otmraz-eu-logs/Resnet/ImageNet/${USER}"
retina_model_dir = "gs://otmraz-eu-logs/Retinanet/${USER}"

prepare_resnet_cmd = '''
preprocessing_file="imagenet_preprocessing.py"
export PYTHONPATH=$HOME/ml_input_processing/experiments/ml/models/
export TF_DUMP_GRAPH_PREFIX=$HOME/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet/graph_dump.log
tpu_name="local"
model_dir="gs://otmraz-eu-logs/Resnet/ImageNet/${USER}"
data_dir="gs://tfdata-imagenet-eu" # This scripts needs 2 subfolders: train, validation
train_epochs=90
'''
prepare_retina_cmd = '''
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
model_dir="gs://otmraz-eu-logs/Retinanet/${USER}"
log_out="main.log"
export PYTHONPATH=$HOME/ml_input_processing/experiments/ml/models/
'''

pecan_cmd = '''
export USE_AUTOORDER=True
export n_loc=10
export DISPATCHER_IP='disp'
log_out="../../../../../../../../pecan-experiments/experiments/pecan/logs/pecan.log"
'''
cachew_cmd = '''
export USE_AUTOORDER=False
export n_loc=0
export DISPATCHER_IP='disp'
log_out="../../../../../../../../pecan-experiments/experiments/pecan/logs/cachew.log"
'''
colloc_cmd = '''
export USE_AUTOORDER=False
export n_loc=0
export DISPATCHER_IP='None'
log_out="../../../../../../../../pecan-experiments/experiments/pecan/logs/colloc.log"
'''

exp_dir = '''../../../../../../../../pecan-experiments/experiments/pecan''' # From resnet dir
resnet_dir = '''../../../ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet/'''
retina_dir = '''../../../ml_input_processing/experiments/ml/models/official/vision/detection/'''

ResNet_cmd = '''python3 resnet_ctl_imagenet_main.py --enable_checkpoint_and_export=true --tpu=$tpu_name --model_dir=$model_dir --data_dir=$data_dir --batch_size=1024 --steps_per_loop=500 --train_epochs=$train_epochs --use_synthetic_data=false --dtype=fp32 --enable_eager=true --enable_tensorboard=true --distribution_strategy=tpu --log_steps=50 --single_l2_loss_op=true --verbosity=0 --skip_eval=true --use_tf_function=true --num_local_workers=$n_loc 2>&1 | tee $log_out'''
ResNet_cmd_param = '''python3 resnet_ctl_imagenet_main.py --enable_checkpoint_and_export=true --tpu=$tpu_name --model_dir=$model_dir --data_dir=$data_dir --batch_size=1024 --steps_per_loop={0} --train_epochs={1} --use_synthetic_data=false --dtype=fp32 --enable_eager=true --enable_tensorboard=true --distribution_strategy=tpu --log_steps=50 --single_l2_loss_op=true --verbosity=0 --skip_eval=true --use_tf_function=true --num_local_workers=$n_loc 2>&1 | tee $log_out'''
Retina_cmd = '''python main.py --strategy_type=tpu --tpu="${TPU_ADDRESS?}" --model_dir="${model_dir?}" --save_checkpoint_freq=$save_checkpoint_freq --mode=train --local_workers=$n_loc --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?}, iterations_per_loop: ${ITERS_PER_LOOP}, total_steps: ${total_steps}}, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?}, num_steps_per_eval: ${ITERS_PER_LOOP} } }" 2>&1 | tee $log_out
'''

cost_extract_cmd = '''python plotting-scripts/extract_costs.py --path={0} --model={1} --accelerator=v2 --experiment_type={2} --header=True'''

plot_cmd = '''python plotting-scripts/fig8.py -e final -m {0} -t {1} -c {2}'''

def get_exitcode_stdout_stderr(cmd):
    # Execute the external command and get its exitcode, stdout and stderr.
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err

def get_disp_ip():
    # Getting the dispatcher ip
    disp_ip_cmd='kubectl get node'
    disp_ip=None
    _, disp_ip_lines, _ = get_exitcode_stdout_stderr(disp_ip_cmd)
    disp_ip_lines = disp_ip_lines.decode('utf-8')
    disp_ip_lines = disp_ip_lines.split('\n')
    for i in disp_ip_lines:
        if 'cachew-dispatcher' in i:
            disp_ip = i.split(' ')[0]
    return disp_ip

def get_costs(extractor_str):
    vals = extractor_str.split('\n')[1]
    tpu_cost = vals.split(',')[2]
    cpu_cost = vals.split(',')[3]
    return tpu_cost, cpu_cost

def set_service_img(img):
    file_lines = []
    with open(service_yaml, 'r') as file:
        for line in file:
            if 'image: "' in line:
                pref = line.split('"')[0]
                suff = line.split('"')[2]
                file_lines.append(pref + '"' + img + '"' + suff)
                #file_lines.append('image: "' + img + '"\n')
            else:
                file_lines.append(line)
    with open(service_yaml, 'w') as f:
        f.write(''.join(file_lines))

disp_ip = get_disp_ip()

### a) Pecan

n_local = 10
n_steps = 5000
set_service_img(pecan_img)

sp.run(restart_workers_cmd, shell=True)
os.chdir(resnet_dir)
sp.run(prepare_resnet_cmd+pecan_cmd+ResNet_cmd_param.format(500, 8), shell=True)
os.chdir(exp_dir)
sp.run('gsutil rm -r '+resnet_model_dir, shell=True)

### b) Cachew

n_local = 0
n_steps = 5000
set_service_img(cachew_img)

sp.run(restart_workers_cmd, shell=True)
os.chdir(resnet_dir)
sp.run(prepare_resnet_cmd+cachew_cmd+ResNet_cmd_param.format(500, 4), shell=True)
os.chdir(exp_dir)
sp.run(stop_workers_cmd, shell=True)
sp.run('gsutil rm -r '+resnet_model_dir, shell=True)

### c) No service

n_local = 0
n_steps = 5000
disp_ip = 'None'

os.chdir(resnet_dir)
sp.run(prepare_resnet_cmd+colloc_cmd+ResNet_cmd_param.format(250, 2), shell=True)
os.chdir(exp_dir)
sp.run('gsutil rm -r '+resnet_model_dir, shell=True)

### d) Plotting
os.chdir(exp_dir)
# TODO: Replace the paths!!!!
_, pecan_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/sample_logs/sample_resnet.log', 'resnet', 'pecan'))
_, cachew_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/sample_logs/resnet_cachew.log', 'resnet', 'cachew'))
_, colloc_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/sample_logs/resnet_colloc.log', 'resnet', 'collocated'))

pecan_tpu, pecan_cpu = get_costs(pecan_out.decode("utf-8"))
cachew_tpu, cachew_cpu = get_costs(cachew_out.decode("utf-8"))
colloc_tpu, colloc_cpu = get_costs(colloc_out.decode("utf-8"))

_, _, _ = get_exitcode_stdout_stderr(plot_cmd.format('ResNet50_v2-8', ' '.join([colloc_tpu, cachew_tpu, pecan_tpu]), ' '.join([colloc_cpu, cachew_cpu, pecan_cpu])))

print("Finished experiments!")