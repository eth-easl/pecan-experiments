Figure 8: Collocated vs. Cachew vs. Pecan
=======

All commands below are intended to be executed on the remote TPU VM as deployed via the `create_tpu_vm.sh` script. If you have not already done so, `ssh` into it using `gcloud alpha compute tpus tpu-vm ssh --zone europe-west4-a atc24-ae-<TPU_name>`

For the initial `kick-the-tires` phase please go through steps 1-6 ("Getting Started" example). For the `full evaluation` of the artifact, please complete steps.

1. **Activate tmux window**. Since some of our experiments can take long to run, make sure to set up a terminal multiplexer such as `tmux` in case your connection is interrupted. A very short introduction can be found at the end of this readme.
2. **Activate the Artifact Eval virtual environment** `source ~/atc_venv/bin/activate`
3. **Go to the experiments directory** `cd ~/pecan-experiments/experiments/pecan`
4. **Starting the cluster**. Execute `./manage_cluster.sh start`. The script will create and setup a cluster of several virtual machines.
5. **Checking the status** of the cluster by executing `./manage_cluster.sh status`. If all the status indicators show a green `[OK]`, carry on with the next step.
6. **Run Hellow World example**. Run the following command to execute a short ResNet run to check the setup is working correctly. `python run_fig8.py -m short` In this run, all workers are used at all times and a single bar is produced in the plot `plots/Getting_started.pdf` (see our Reference Results below and the sample log file `logs/sample_logs/hello_world.log`).
7. **Run ResNet50 experiment**. Start up the cluster again (steps 3. and 4.). Run the following command to execute the experiments for the ResNet in Fig 8. `python run_fig8.py -m ResNet50_v2-8`. More concretely this script does the following:
    a. **Running the input pipeline with Pecan** - producing data for the brown bars
    b. **Running the input pipeline with Cachew** - producing data for the orange bars
    c. **Running the pipeline in the collocated mode** - producing data for the blue bars
    d. **Generating a plot with the cost of each config** - generating a plot `plots/fig8_ResNet50_v2-8.pdf`
8. **Run RetinaNet experiment**. Start up the cluster again (steps 3. and 4.). Then run the following command to exectue the experiments for the RetinaNet in Fig 8. `python run_fig8.py -m retina`. It runs the same experiments as for ResNet.
9. **Tear down the cluster**. Please make sure to execute `./manage_cluster.sh stop` to tear down the cluster. All statuses should show a green `[OK]`.

We use Google Cloud Compute Engine for all experiments and read input datasets from Google Cloud Storage buckets. Please see [this](https://docs.google.com/spreadsheets/d/1iwkurV_3AxQ7a_KcKKhgDBbO5r0rSQZxcjTqwgxE9Mg/edit?usp=sharing) spreadsheet for an estimate of the time and cost of running each of the above experiments.

*How to use `tmux`*: Execute `tmux` in the current directory. Then whatever command you want to execute in the background (i.e. `./experiment_fig7.sh`). You may now close this window by actually closing the terminal itself, **do not use `Ctrl+C` / `Ctrl+D`**. If at a later point you would like to check in on the experiment, ssh into your machine and execute `tmux attach -t 0` (tmux supports multiple of those "background sessions", so if you have multiple open sessions, you may be looking for an integer larger than `0`). In general you may want to interact with `tmux` using [keyboard shortcuts](https://gist.github.com/MohamedAlaa/2961058).

### Reference Results and Variability

This experiment runs Cachew's AutoScaling and Pecan's AutoOrder & AutoPlacement policices. Both AutoScaling & AutoPlacement rely on (somewhat noisy) runtime metrics. The permormance is also impacted by the current load in Google Cloud. Hence it is expected to observe some variability in the exact costs of Cachew and Pecan, in particular the remote worker costs.

**The key result here is that Pecan incurrs a significantly lower cost than collocated or Cachew setups.**

Below we offer the reference results:

Hellow World experiment:

<img src="../pecan/plots/sample_plots/Getting_started.jpg" height=240/>

ResNet experiment:

<img src="../pecan/plots/sample_plots/fig8_ResNet50_v2-8.jpg" height=240/>

RetinaNet experiment:

<img src="../pecan/plots/sample_plots/fig8_RetinaNet.jpg" height=240/>

See also the `logs/sample_logs` directory for example logs of the experimentss