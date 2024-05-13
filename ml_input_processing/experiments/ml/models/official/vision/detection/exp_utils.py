# Helper code to get dispatcher (to use inside pipelines)

import subprocess as sp
from subprocess import Popen, PIPE
import shlex

def get_exitcode_stdout_stderr(cmd):
    # Execute the external command and get its exitcode, stdout and stderr.
    args = shlex.split(cmd)
    proc = Popen(args, stdout=PIPE, stderr=PIPE)
    out, err = proc.communicate()
    exitcode = proc.returncode
    return exitcode, out, err

#DISPATCHER_IP = 'ascascasc'

# Will try to get a dispatcher spawned from the same client
# If unsuccessful, will look at whole gcloud project for any usable dispatcher
def get_disp_ip(DISPATCHER_IP):

    if DISPATCHER_IP is not None:
        # Getting the dispatcher ip
        disp_ip_cmd='kubectl get node'
        disp_ip=None
        _, disp_ip_lines, _ = get_exitcode_stdout_stderr(disp_ip_cmd)
        disp_ip_lines = disp_ip_lines.decode('utf-8')
        disp_ip_lines = disp_ip_lines.split('\n')
        for i in disp_ip_lines:
            if 'cachew-dispatcher' in i:
                disp_ip = i.split(' ')[0]

        # If there is no dispatcher spawned from the current Client VM, we might be able to get it
        #print(disp_ip)
        #print(disp_ip is None)
        if disp_ip is None:
            print("empty")
            disp_ip_cmd='gcloud compute instances list'
            _, disp_ip_lines, _ = get_exitcode_stdout_stderr(disp_ip_cmd)
            disp_ip_lines = disp_ip_lines.decode('utf-8')
            disp_ip_lines = disp_ip_lines.split('\n')
            for i in disp_ip_lines:
                if 'cachew-dispatcher' in i:
                    DISPATCHER_IP = i.split(' ')[0]
        else:
            DISPATCHER_IP=disp_ip

    return DISPATCHER_IP
