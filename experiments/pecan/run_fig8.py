import os, sys
import subprocess as sp
from subprocess import Popen, PIPE
import shlex
import fileinput


'''
    a. **Running the input pipeline with Pecan** - producing data for the brown bars
    b. **Runnign the input pipeline with Cachew** - producing data for the orange bars
    c. **Runing the pipeline in the collocated mode** - producing data for the blue bars
    d. **Generating a plot with the cost of each config**. generating a plot `fig8_ResNet50_v2-8.pdf`
'''

restart_workers_cmd = '''./manage_cluster restart_service'''
stop_workers_cmd = '''./manage_cluster stop'''
service_yaml = '''default_config.yaml'''

pecan_img = '''tf_oto:pecan'''
cachew_img = '''tf_oto:pecan''' # Same as Pecan, we just don't spawn any local workers

prepare_resnet_cmd = '''
cd ~/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet

preprocessing_file="imagenet_preprocessing.py"
logging_file="exp_log"+time.strftime("%Y_%m_%d-%H_%M_%S")+".txt"

export PYTHONPATH=$HOME/ml_input_processing/experiments/ml/models/
export TF_DUMP_GRAPH_PREFIX=$HOME/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet/graph_dump.log

tpu_name="local"
model_dir="gs://otmraz-eu-logs/Resnet/ImageNet/FINAL"
data_dir="gs://tfdata-imagenet-eu" # This scripts needs 2 subfolders: train, validation

train_epochs=90"
'''

pecan_cmd = '''
export USE_AUTOORDER=True
export n_loc=10
export DISPATCHER_IP='disp'
log_out="pecan.log
'''
cachew_cmd = '''
export USE_AUTOORDER=False
export n_loc=0
export DISPATCHER_IP='disp'
log_out="cachew.log
'''
colloc_cmd = '''
export USE_AUTOORDER=False
export n_loc=10
export DISPATCHER_IP='None'
log_out="colloc.log
'''


ResNet_cmd = '''
python3 resnet_ctl_imagenet_main.py --enable_checkpoint_and_export=true --tpu=$tpu_name --model_dir=$model_dir --data_dir=$data_dir --batch_size=1024 --steps_per_loop=50 --train_epochs=$train_epochs --use_synthetic_data=false --dtype=fp32 --enable_eager=true --enable_tensorboard=true --distribution_strategy=tpu --log_steps=50 --single_l2_loss_op=true --verbosity=0 --skip_eval=true --use_tf_function=true --num_local_workers=$n_loc 2>&1 | tee $log_out
'''

Retina_cmd = '''

'''


output_fig = '''fig8_ResNet50_v2-8.pdf'''

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

disp_ip = get_disp_ip()
_, _, _ = get_exitcode_stdout_stderr(prepare_resnet_cmd)

### a) Pecan

n_local = 10
n_steps = 5000



# Set correct service img
file_lines = []
with open(service_yaml, 'r') as file:
    for line in file:
        if 'image: "' in line:
            file_lines.append('image: "' + 'tf_oto:pecan')
        else:
            file_lines.append(line)



_, _, _ = get_exitcode_stdout_stderr(restart_workers_cmd)
_, _, _ = get_exitcode_stdout_stderr(pecan_cmd)
_, _, _ = get_exitcode_stdout_stderr(ResNet_cmd)



### b) Cachew

n_local = 0
n_steps = 5000

# Set correct service img
file_lines = []
with open(service_yaml, 'r') as file:
    for line in file:
        if 'image: "' in line:
            file_lines.append('image: "' + 'tf_oto:pecan')
        else:
            file_lines.append(line)



_, _, _ = get_exitcode_stdout_stderr(restart_workers_cmd)
_, _, _ = get_exitcode_stdout_stderr(cachew_cmd)
_, _, _ = get_exitcode_stdout_stderr(ResNet_cmd)
_, _, _ = get_exitcode_stdout_stderr(stop_workers_cmd)


### c) No service

n_local = 0
n_steps = 5000
disp_ip = 'None'

# Set correct service img
file_lines = []
with open(service_yaml, 'r') as file:
    for line in file:
        if 'image: "' in line:
            file_lines.append('image: "' + 'tf_oto:pecan')
        else:
            file_lines.append(line)

_, _, _ = get_exitcode_stdout_stderr(colloc_cmd)
_, _, _ = get_exitcode_stdout_stderr(ResNet_cmd)


### d) Plotting

