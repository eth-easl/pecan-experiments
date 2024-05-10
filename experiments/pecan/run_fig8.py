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

ResNet_cmd = '''
FIXMEEEEEEE {0} FIXMEEE {1}
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



sp.run(restart_workers_cmd)

sp.run(ResNet_cmd.format(n_local, n_steps))



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



sp.run(restart_workers_cmd)

sp.run(ResNet_cmd.format(n_local, n_steps))

sp.run(stop_workers_cmd)


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



sp.run(restart_workers_cmd)

sp.run(ResNet_cmd.format(n_local, n_steps))


### d) Plotting

