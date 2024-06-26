import os
import subprocess as sp
from subprocess import Popen, PIPE
import shlex
import argparse

parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('-m', '--model', type=str, help='model used', default='ResNet50_v2-8')

args = parser.parse_args()

model = args.model

# a. **Run the input pipeline with Pecan** - producing data for the brown bars
# b. **Run the input pipeline with Cachew** - producing data for the orange bars
# c. **Run the pipeline in the collocated mode** - producing data for the blue bars
# d. **Generate a plot with the cost of each config**. generating a plot, e.g., `fig8_ResNet50_v2-8.pdf`

restart_workers_cmd = '''./manage_cluster.sh restart_service'''
stop_workers_cmd = '''./manage_cluster.sh stop'''
service_yaml = '''default_config.yaml'''

pecan_img = '''tf_oto:dan_fast_removal_ae''' #'''tf_oto:pecan'''
pecan_ae_img = '''tf_oto:dan_fast_removal_ae'''
cachew_img = '''tf_oto:pecan''' # We use the pecan img, but we disable autoorder and simply spawn 0 remote workers

resnet_model_dir = "gs://otmraz-eu-logs/Resnet/ImageNet/${USER}"
retina_model_dir = "gs://otmraz-eu-logs/Retinanet/${USER}"

prepare_resnet_cmd = '''
preprocessing_file="imagenet_preprocessing.py"
export PYTHONPATH=$HOME/pecan-experiments/ml_input_processing/experiments/ml/models/
export TF_DUMP_GRAPH_PREFIX=$HOME/pecan-experiments/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet/graph_dump.log
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
epochs=$num_epochs # default: 13, 72 is 2nd version
total_steps=$(($epochs * $ITERS_PER_LOOP))
eval=false
save_checkpoint_freq=1
model_dir="gs://otmraz-eu-logs/Retinanet/${USER}"
export PYTHONPATH=$HOME/pecan-experiments/ml_input_processing/experiments/ml/models/
'''

getting_started_cmd = '''
export USE_AUTOORDER=True
export n_loc=10
export DISPATCHER_IP='disp'
log_out="../../../../../../../../../pecan-experiments/experiments/pecan/logs/resnet_getting_started.log"
'''
pecan_cmd = '''
export USE_AUTOORDER=True
export n_loc=10
export DISPATCHER_IP='disp'
log_out="../../../../../../../../../pecan-experiments/experiments/pecan/logs/resnet_pecan.log"
'''
cachew_cmd = '''
export USE_AUTOORDER=False
export n_loc=0
export DISPATCHER_IP='disp'
log_out="../../../../../../../../../pecan-experiments/experiments/pecan/logs/resnet_cachew.log"
'''
colloc_cmd = '''
export USE_AUTOORDER=False
export n_loc=0
export DISPATCHER_IP='None'
log_out="../../../../../../../../../pecan-experiments/experiments/pecan/logs/resnet_colloc.log"
'''

pecan_retina_env_file_path = 'env/pecan_retina_env.sh'
cachew_retina_env_file_path = 'env/cachew_retina_env.sh'
colloc_retina_env_file_path = 'env/colloc_retina_env.sh'
pecan_retina_cmd = '''
export USE_AUTOORDER=True
export n_loc=10
export DISPATCHER_IP='disp'
export num_epochs=7
export log_out="../../../../../../../../pecan-experiments/experiments/pecan/logs/retina_pecan.log"
'''
cachew_retina_cmd = '''
export USE_AUTOORDER=False
export n_loc=0
export DISPATCHER_IP='disp'
num_epochs=4
log_out="../../../../../../../../pecan-experiments/experiments/pecan/logs/retina_cachew.log"
'''
colloc_retina_cmd = '''
export USE_AUTOORDER=False
export n_loc=0
export DISPATCHER_IP='None'
num_epochs=2
log_out="../../../../../../../../pecan-experiments/experiments/pecan/logs/retina_colloc.log"
'''

exp_dir = '''../../../../../../../../../pecan-experiments/experiments/pecan''' # From resnet dir
exp_dir_from_retina = '''../../../../../../../../pecan-experiments/experiments/pecan''' # From resnet dir
resnet_dir = '''../../ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet/'''
retina_dir = '''../../ml_input_processing/experiments/ml/models/official/vision/detection/'''

ResNet_cmd_param = '''python3 resnet_ctl_imagenet_main.py --enable_checkpoint_and_export=true --tpu=$tpu_name --model_dir=$model_dir --data_dir=$data_dir --batch_size=1024 --steps_per_loop={0} --train_epochs={1} --use_synthetic_data=false --dtype=fp32 --enable_eager=true --enable_tensorboard=true --distribution_strategy=tpu --log_steps=50 --single_l2_loss_op=true --verbosity=0 --skip_eval=true --use_tf_function=true --num_local_workers=$n_loc 2>&1 | tee $log_out'''
retina_cmd_param = '''python main.py --strategy_type=tpu --tpu="${TPU_ADDRESS?}" --model_dir="${model_dir?}" --save_checkpoint_freq=$save_checkpoint_freq --mode=train --local_workers=$n_loc --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?}, iterations_per_loop: 250, total_steps: ${total_steps}}, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?}, num_steps_per_eval: ${ITERS_PER_LOOP} } }" 2>&1 | tee $log_out'''

cost_extract_cmd = '''python plotting-scripts/extract_costs.py --path={0} --model={1} --accelerator=v2 --experiment_type={2} --header=True'''

plot_cmd = '''python plotting-scripts/fig8.py -e final -m {0} -t {1} -c {2} -o {3}'''

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
            else:
                file_lines.append(line)
    with open(service_yaml, 'w') as f:
        f.write(''.join(file_lines))

def set_scaling_policy(scaling_policy):
    file_lines = []
    with open(service_yaml, 'r') as file:
        for line in file:
            if 'scaling_policy: ' in line:
                pref = line.split(' ')[0]
                suff = line.split(' ')[1]
                suff = str(scaling_policy) + suff[1:]
                file_lines.append(pref + ' ' + suff)
            else:
                file_lines.append(line)
    with open(service_yaml, 'w') as f:
        f.write(''.join(file_lines))

disp_ip = get_disp_ip()

if model == 'short':
    print('Running Getting Started experiment')

    ### a) Pecan (with all workers used all the time)
    sp.run('gsutil rm -r '+resnet_model_dir, shell=True)
    set_service_img(pecan_img)
    set_scaling_policy(2) # Fixed number of workers

    sp.run(restart_workers_cmd, shell=True)
    set_scaling_policy(3) # Fix the policy back for the 'real' experiments
    os.chdir(resnet_dir)
    sp.run(prepare_resnet_cmd+getting_started_cmd+ResNet_cmd_param.format(500, 2), shell=True)
    os.chdir(exp_dir)
    sp.run(stop_workers_cmd, shell=True)
    sp.run('gsutil rm -r '+resnet_model_dir, shell=True)

    ### d) Plotting
    _, pecan_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/resnet_getting_started.log', 'resnet', 'pecan'))
    pecan_tpu, pecan_cpu = get_costs(pecan_out.decode("utf-8"))

    sp.run(plot_cmd.format('ResNet50_v2-8', ' '.join(['0.0', '0.0', pecan_tpu]), ' '.join(['0.0', '0.0', pecan_cpu]), 'plots/Getting_started'), shell=True)

if model == 'ResNet50_v2-8':

    print('Running Resnet experiments')
    ### a) Pecan
    sp.run('gsutil rm -r '+resnet_model_dir, shell=True)
    set_service_img(pecan_ae_img)

    sp.run(restart_workers_cmd, shell=True)
    os.chdir(resnet_dir)
    sp.run(prepare_resnet_cmd+pecan_cmd+ResNet_cmd_param.format(500, 8), shell=True)
    os.chdir(exp_dir)
    sp.run('gsutil rm -r '+resnet_model_dir, shell=True)

    ### b) Cachew
    set_service_img(cachew_img)

    sp.run(restart_workers_cmd, shell=True)
    os.chdir(resnet_dir)
    sp.run(prepare_resnet_cmd+cachew_cmd+ResNet_cmd_param.format(500, 4), shell=True)
    os.chdir(exp_dir)
    sp.run(stop_workers_cmd, shell=True)
    sp.run('gsutil rm -r '+resnet_model_dir, shell=True)

    ### c) No service
    disp_ip = 'None'

    os.chdir(resnet_dir)
    sp.run(prepare_resnet_cmd+colloc_cmd+ResNet_cmd_param.format(250, 2), shell=True)
    os.chdir(exp_dir)
    sp.run('gsutil rm -r '+resnet_model_dir, shell=True)

    ### d) Plotting
    _, pecan_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/resnet_pecan.log', 'resnet', 'pecan'))
    _, cachew_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/resnet_cachew.log', 'resnet', 'cachew'))
    _, colloc_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/resnet_colloc.log', 'resnet', 'collocated'))

    pecan_tpu, pecan_cpu = get_costs(pecan_out.decode("utf-8"))
    cachew_tpu, cachew_cpu = get_costs(cachew_out.decode("utf-8"))
    colloc_tpu, colloc_cpu = get_costs(colloc_out.decode("utf-8"))

    sp.run(plot_cmd.format('ResNet50', ' '.join([colloc_tpu, cachew_tpu, pecan_tpu]), ' '.join([colloc_cpu, cachew_cpu, pecan_cpu]), 'plots/ResNet50_v2-8'), shell=True)

elif model == 'retina':
    print('Running Retina experiments')

    sp.call(['bash', 'retina.sh'])
    sp.run('gsutil rm -r '+retina_model_dir, shell=True)

    ### d) Plotting
    _, pecan_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/retina_pecan.log', 'retinanet', 'pecan'))
    _, cachew_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/retina_cachew.log', 'retinanet', 'cachew'))
    _, colloc_out, _ = get_exitcode_stdout_stderr(cost_extract_cmd.format('logs/retina_colloc.log', 'retinanet', 'collocated'))

    pecan_tpu, pecan_cpu = get_costs(pecan_out.decode("utf-8"))
    cachew_tpu, cachew_cpu = get_costs(cachew_out.decode("utf-8"))
    colloc_tpu, colloc_cpu = get_costs(colloc_out.decode("utf-8"))

    sp.run(plot_cmd.format('RetinaNet', ' '.join([colloc_tpu, cachew_tpu, pecan_tpu]), ' '.join([colloc_cpu, cachew_cpu, pecan_cpu]), 'plots/RetinaNet'), shell=True)

print("Finished experiments!")