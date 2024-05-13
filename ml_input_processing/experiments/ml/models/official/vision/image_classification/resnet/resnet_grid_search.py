import subprocess
import subprocess as sp
from subprocess import Popen, PIPE
import shlex
import numpy
import os, time, re
import argparse

from official.vision.image_classification.resnet.exp_utils import get_disp_ip

parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

parser.add_argument('-rw', '--remote_workers', type=int, help='the number of remote workers used', default=0)
args = parser.parse_args()

local_worker_max = 10
no_rem_workers = args.remote_workers

# Currently we will use 1,2,3,...,10 and 20 local workers (as in Pecan paper)
loc_worker_counts = [i for i in range(local_worker_max)]
if no_rem_workers == 0:
    loc_worker_counts = [i for i in range(1, local_worker_max)]
loc_worker_counts.append(local_worker_max)
loc_worker_counts.append(20)

def get_exitcode_stdout_stderr(cmd):
    # Execute the external command and get its exitcode, stdout and stderr.
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err

# Set args
#args_cmd='''
# preprocessing_file = "imagenet_preprocessing.py"
logging_file = "exp_log" + time.strftime("%Y_%m_%d-%H_%M_%S") + ".txt"

tpu_name="local"
model_dir="gs://otmraz-eu-logs/Resnet/ImageNet/LW"
data_dir="gs://tfdata-imagenet-eu" # This scripts needs 2 subfolders: train, validation

train_epochs="1" # At least 5 for CIFAR10
log_out="resnet_test.log" # To folder with timestamp!

# The grid search is ALWAYS done using a disaggregated preproc service
disp_ip = get_disp_ip('disp')
print("Dispatcher name is: " + str(disp_ip))

# Replacing the dispatcher ip in the relevant files
'''with open('imagenet_preprocessing.py', 'r') as file :
  filedata = file.read()


filedata = filedata.split('\n')
for i in range(len(filedata)):
    if '-cachew-dispatcher-' in filedata[i]:
        filedata[i] = "DISPATCHER_IP='" + disp_ip + "'"
    elif filedata[i] == "DISPATCHER_IP=None":
        filedata[i] = "#DISPATCHER_IP=None"
    # We don't use any reordering in the experiment
    elif 'keep_position=False' in filedata[i]:
        filedata[i] = filedata[i].replace('keep_position=False', 'keep_position=True')


filedata = '\n'.join(filedata)

# Write the file out again
with open('new_imagenet_preprocessing.py', 'w') as file:
  file.write(filedata)
'''

d = dict(os.environ)
d["PYTHONPATH"] = '''/home/otomr/ml_input_processing/experiments/ml/models/'''
sp.Popen('/bin/echo $PYTHONPATH', shell=True, env=d).wait()
d["TF_DUMP_GRAPH_PREFIX"] = '''/home/otomr/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet/graph_dump.log'''
sp.Popen('/bin/echo $TF_DUMP_GRAPH_PREFIX', shell=True, env=d).wait()



for i in loc_worker_counts:

    if no_rem_workers != 0:

        os.chdir(os.getenv("HOME"))
        os.chdir("cachew_experiments/experiments/autocaching")
        #subprocess.run("./manage_cluster.sh restart_service")
        get_exitcode_stdout_stderr("./manage_cluster.sh restart_service")

        os.chdir(os.getenv("HOME"))
        os.chdir("ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet")
    elif i == 0: 
        pass

    out_file = "resnet_test_"+str(no_rem_workers)+"_"+str(i)+".log"
    model_dir = "gs://otmraz-eu-logs/Resnet/ImageNet/LW_"+str(no_rem_workers)+"_"+str(i)
    #subprocess.run(out_file)
    #subprocess.run(model_dir)

    d["out_file"] = out_file
    sp.Popen('/bin/echo $out_file', shell=True, env=d).wait()
    d["model_dir"] = model_dir
    sp.Popen('/bin/echo $model_dir', shell=True, env=d).wait()

    # Run the actual training
    train_cmd='''python3 resnet_ctl_imagenet_main.py --dispatcher_ip={0} --num_local_workers={1} --enable_checkpoint_and_export=true --tpu={2} \
        --model_dir={3}   --data_dir={4}   --batch_size=1024   --steps_per_loop=250   --train_epochs={5}   --use_synthetic_data=false \
        --dtype=fp32   --enable_eager=true   --enable_tensorboard=true   --distribution_strategy=tpu   --log_steps=50 \
        --single_l2_loss_op=true   --verbosity=0 --skip_eval=true --use_tf_function=true 2>&1 | tee {6}
    '''.format(disp_ip, i, tpu_name, model_dir, data_dir, train_epochs, out_file)
    #subprocess.run(train_cmd.format(i))
    sp.Popen(train_cmd, shell=True, env=d).wait()
    #get_exitcode_stdout_stderr(train_cmd)

os.chdir(os.getenv("HOME"))
os.chdir("cachew_experiments/experiments/autocaching")
get_exitcode_stdout_stderr("./manage_cluster.sh stop")
