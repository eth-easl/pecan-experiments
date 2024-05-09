All of the below commands are intended to be executed on the remote TPU VM as deployed via the `deploy_vm.sh` script.

1. **Go to the experiment directory** if you haven't already `cd ~/experiments/pecan`
2. **Activate tmux window**. Since some of our experiments can take long to run, make sure to setup a terminal multiplexer such as `tmux` in case your connection is interrupted. A very short introduction can be found at the end of this readme.
3. **Starting the cluster**. Execute `./manage_cluster.sh start`. The script will create and setup a cluster of several virtual machines.
4. **Checking the status** of the cluster by executing `./manage_cluster.sh status`. If all the status indicators show a green `[OK]`, carry on with the next step.
5. **Run ResNet50 experiment**. Run the following command to exectue the experiments for the ResNet in Fig 8. `TODO: Prepare python script for a 1 line effor and cmd`. More concretely this script does the following:
    a. **Running the input pipeline with Pecan** - producing data for the brown bars
    b. **Runnign the input pipeline with Cachew** - producing data for the orange bars
    c. **Runing the pipeline in the collocated mode** - producing data for the blue bars
    d. **Generating a plot with the cost of each config**. generating a plot `fig8_ResNet50_v2-8.pdf`
6. **Tear down the cluster**. Please make sure to execute `./manage_cluster.sh stop` to tear down the cluster. All statuses should show a green `[OK]`.

*How to use `tmux`*: Execute `tmux` in the current directory. Then whatever command you want to execute in the background (i.e. `./experiment_fig7.sh`). You may now close this window by actually closing the terminal itself, **do not use `Ctrl+C` / `Ctrl+D`**. If at a later point you would like to check in on the experiment, ssh into your machine and execute `tmux attach -t 0` (tmux supports multiple of those "background sessions", so if you have multiple open sessions, you may be looking for an integer larger than `0`). In general you may want to interact with `tmux` using [keyboard shortcuts](https://gist.github.com/MohamedAlaa/2961058).

### Reference Result and Variability

This experiment runs Cachew's AutoScaling and Pecan's AutoOrder & AutoPlacement policices. Since both AutoScaling & AutoPlacement rely on (somewhat noisy) runtime metrics, expect some variability in the exact costs of Cachew and Pecan, in particular the remote worker costs. Also note, due to the artifact evaluation time constrains, we are not running the AutoPlacement-only experiment.

Below we offer the reference result (TODO: update):

<img src="reference-results/ref.png" height=480/>