# Pecan: Cost-Efficient ML Data Preprocessing with Automatic Transformation Ordering and Hybrid Placement

Pecan is a multi-tenant service for cost-efficient input data processing in machine learning jobs. 

To minimize end-to-end training cost, Pecan uses 2 policies: 
1) AutoPlacement, which distributes data prococessing workers between the client VM and additional remote worker nondes.
2) AutoOrder, which reorders non-critical step in the input pipeline in order to minimize the overall required CPU cycles and increase throughput.

Pecan builds on top of the [cachew](https://www.usenix.org/system/files/atc22-graur.pdf) data loading framework in [TensorFlow](https://github.com/tensorflow/tensorflow), extending it with the AutoPlacement and AutoOrder policies.

This repository contains instructions for deploying Pecan in Google Cloud and using the service to efficiently execute ML input data pipelines. To view the source code, please see our [Pecan source code repository](https://github.com/eth-easl/cachew/tree/oto-pecan). 

__Should we make a separate Pecan repo as well rather than have it as a branch in Cachew?__
