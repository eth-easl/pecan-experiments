Pecan experiments
=======

This folder provides scripts and instructions for reproducing key results from our USENIX ATC'24 paper:

1. Fig.8 Reproducing the ResNet50_v2-8 bars. In this experiment we compare the performance of collocated data preprocessing (blue bars), Cachew (orange bars), and Pecan (brown bars).

We use Google Cloud Compute Engine for all experiments and read input datasets from Google Cloud Storage buckets. Please see [this](https://docs.google.com/spreadsheets/d/1iwkurV_3AxQ7a_KcKKhgDBbO5r0rSQZxcjTqwgxE9Mg/edit?usp=sharing) spreadsheet for an estimate of the time and cost of running each of the above experiments.

In order to execute the experiments in a reasonable time, we enable the fast worker-removal feature in these experiments as described in Section 5.1.

## Troubleshooting

### Failed to Mount GlusterFS

During the deployment of a cluster using the `./manage_cluster.sh start` command (or any of its variants), GlusterFS is being deployed. It can sometimes happen that GlusterFS deployment will fail due to some reason. This will show up in the deployment check (note that GlusterFS could not be mounted):

<img src="../failed_gluster.png" height=240/>

In such cases, you should terminate the deployment via `./manage_cluster.sh stop` and try to redeploy via `./manage_cluster.sh start` .
