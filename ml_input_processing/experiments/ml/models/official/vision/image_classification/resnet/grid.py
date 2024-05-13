import subprocess
import subprocess as sp
from subprocess import Popen, PIPE
import shlex
import numpy
import os, time

local_worker_max = 10
no_rem_workers = 12

loc_worker_counts = [i for i in range(1, local_worker_max)]
loc_worker_counts.append(local_worker_max)
loc_worker_counts.append(20)
#loc_worker_counts = [*range(0,local_worker_max+1, 1)].append(20)

def get_exitcode_stdout_stderr(cmd):
    """
    Execute the external command and get its exitcode, stdout and stderr.
    """
    args = shlex.split(cmd)

    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    #
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
#'''

#subprocess.run(args_cmd)
#get_exitcode_stdout_stderr(args_cmd)

#get_exitcode_stdout_stderr('''export PYTHONPATH=/home/otomr/ml_input_processing/experiments/ml/models/''')
#sp.Popen('/bin/echo $PYTHONPATH', shell=True, env=d).wait()

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
    train_cmd='''python3 resnet_ctl_imagenet_main.py --num_local_workers={0} --enable_checkpoint_and_export=true --tpu={1} \
        --model_dir={2}   --data_dir={3}   --batch_size=1024   --steps_per_loop=50   --train_epochs={4}   --use_synthetic_data=false \
        --dtype=fp32   --enable_eager=true   --enable_tensorboard=true   --distribution_strategy=tpu   --log_steps=50 \
        --single_l2_loss_op=true   --verbosity=0 --skip_eval=true --use_tf_function=true 2>&1 | tee {5}
    '''.format(i, tpu_name, model_dir, data_dir, train_epochs, out_file)
    #subprocess.run(train_cmd.format(i))
    sp.Popen(train_cmd, shell=True, env=d).wait()
    #get_exitcode_stdout_stderr(train_cmd)

os.chdir(os.getenv("HOME"))
os.chdir("cachew_experiments/experiments/autocaching")
get_exitcode_stdout_stderr("./manage_cluster.sh stop")
