# Pecan: Cost-Efficient ML Data Preprocessing with Automatic Transformation Ordering and Hybrid Placement

Pecan is a multi-tenant service for cost-efficient input data processing in machine learning jobs. 

To minimize end-to-end training cost, Pecan uses 2 policies: 
1) AutoPlacement, which distributes data prococessing workers between the client VM and additional remote worker nondes.
2) AutoOrder, which reorders non-critical step in the input pipeline in order to minimize the overall required CPU cycles and increase throughput.

Pecan builds on top of the [cachew](https://www.usenix.org/system/files/atc22-graur.pdf) data loading framework in [TensorFlow](https://github.com/tensorflow/tensorflow), extending it with the AutoPlacement and AutoOrder policies.

This repository contains instructions for deploying Pecan in Google Cloud and using the service to efficiently execute ML input data pipelines. To view the source code, please see our [Pecan source code repository](https://github.com/eth-easl/cachew/tree/oto-pecan). 

__Should we make a separate Pecan repo as well rather than have it as a branch in Cachew?__

## Pecan System Architecture

Pecan consists of a centralized dispatcher, a dynamic number of remote input data workers, and a disaggregated storage cluster for data caching.

![cachew-system-architecture](Figures/pecan)

Users register training nodes (i.e. clients) with the Pecan dispatcher. To execute an input pipeline with Pecan, clients provide a graph representation of the input pipeline and a path to the input dataset in a cloud storage bucket. Pecan supports and extends the tf.data API for defining input data pipelines from a collection of composable and user-parametrizable operators.

TODO:
Users can annotate their tf.data input pipeline to mark candidate locations for caching/reusing data across executions. Pecan will automatically apply caching at the throughput-optimal location in the input pipeline among the candidate locations. 

Cachew's input data workers are stateless components responsible for producing batches of preprocessed data for clients. The dispatcher dynamically adjusts the number of input data workers for each job to minimize epoch time while keeping costs low. The dispatcher also profiles and maintains metadata about input pipeline executions across jobs to make data caching decisions. Cachew stores cached datasets in a GlusterFS remote storage cluster. 

Clients fetch data from the workers that are assigned to them by the dispatcher. Clients and workers periodically send heartbeats to the dispatcher to maintain membership in the service and provide metrics used for the autoscaling and autocaching policies.


## <a name="prerequisites"/>Prerequisites

### General Prerequisites

Our scripts make extensive use of the `gcloud CLI` tool. As a consequence, this tool is a prerequisite for setting up VMs and running experiments. Please follow [this tutorial](https://cloud.google.com/sdk/docs/install) to install it. We additionally make use of the `gsutil` tool. To install it, please follow [this tutorial](https://cloud.google.com/storage/docs/gsutil_install). We also suggest that you use Python 3.9 when using Cachew. In this sense we recommend [PyEnv](https://github.com/pyenv/pyenv) as a means to install and manage multiple python versions and virtual environments.


### Software Prerequisites for Full Service Deployment

If you plan to deploy Cachew as a full service, you will need to set up a client VM which meets the following software dependencies:

* Ubuntu 18.04.5 LTS (GNU/Linux 5.4.0-1072-gcp x86\_64) with root access
* kubectl v1.21.3
* kops v.1.20
* Nvidia GPU Driver v460.27.04
* CUDA v11.2
* cuDNN v8.1
* Python 3.9.12
* Google Cloud SDK (preferably v384.0.0)


To deploy the service itself, one requires 
* A Docker image deploying Cachew builds with CPU-only support. This is used in the Cachew service for the Dispatcher and Workers
* A client-only build of Cachew with GPU/TPU support. 

A safe commit hash at which these can be built is `c7b02e90b4384e721f7c6b13ec55a21cd5295a47`.

### Hardware Prerequisites for Full Service Deployment

If you plan to deploy a Full Cachew Service, you will need the following hardware for your Client VM:

* Intel or AMD x86 CPU with hardware virtualization support
* Nvidia V100 GPUs or v3-8 TPUs
* Around 50 GB of disk space on your root partition

For the Dispatcher as well as the Worker nodes, one requires only VMs with compute power. No accelerators are required. 

### Deployment and Experiment Automation

Since deploying a cluster and running experiments can be complicated, we provide a set of scripts which automate these processes. For deploying a client VM you can use the scripts in the [deploy](deploy). Scripts for running artifact evaluations are found in [experiments](experiments). Further information on how to use this is provided in [this section](#artifact_eval).









## Building Pecan

**Please note that you are not required to build Pecan or generate Docker images for it, as we have pre-built all the necessary binaries for running the artifact evaluation experiments.** We do however, provide scripts for building Pecan and generating its images. These can be found in the [build](build) folder. For more details, please follow the README file in the aforementioned directory.

## Contributing

We welcome contributions to Pecan. Please see our [Cachew source code](https://github.com/eth-easl/cachew/tree/oto-pecan) repository.

 
## Referencing our work

Pecan will appear at USENIX ATC'22. If you decide to use Pecan in your work, please cite our paper: 

```
@inproceedings{cachew,
  author    = {Dan-Ovidiu Graur and
               Oto Mraz and
               Muyu Li and
               Sepehr Pourghannad and
               Chandramohan A. Thekkath and
               Ana Klimovic},
  title     = {Pecan: Cost-Efficient ML Data Preprocessing with Automatic Transformation Ordering and Hybrid Placement},
  booktitle = {Proceedings of the USENIX Annual Technical Confernece (ATC'24)},
  publisher = {{USENIX}},
  year      = {2024},
}
```

