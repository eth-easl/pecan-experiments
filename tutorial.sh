
VM_NAME="atc24-ae-${USER}"

# Create the VM
while ! gcloud alpha compute tpus tpu-vm create --zone=europe-west4-a --accelerator-type="v2-8" --version='tpu-vm-tf-2.8.0' ${VM_NAME}; do echo "Trying again..."; done

# SSH into the VM
gcloud alpha compute tpus tpu-vm ssh --zone europe-west4-a ${VM_NAME}

# Set up the python environment
alias python=python3 >> .bashrc
source .bashrc

apt install python3.8-venv
python -m venv atc_venv
source atc_venv/bin/activate

# Set up the python environment
pip install wheel
gsutil gs://easl-atc24-ae-files/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl .

# Clone the relevant repository
git clone -b main https://github.com/eth-easl/pecan-experiments.git

python -m pip install -r pecan-experiments/requirements.txt
python -m pip install --force-reinstall tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl

## Install other dependencies

# Fix GCP TPU-VM bugs
curl -O https://packages.cloud.google.com/apt/doc/apt-key.gpg 
sudo apt-key add apt-key.gpg
sudo apt update -y && sudo apt-get update

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client

# kops
curl -Lo kops https://github.com/kubernetes/kops/releases/download/v1.19.3/kops-linux-amd64
chmod +x ./kops
sudo mv ./kops /usr/local/bin/

# jq
sudo apt-get install jq

# GlusterFS
sudo add-apt-repository ppa:gluster/glusterfs-9
sudo apt update
sudo apt install -y glusterfs-client
sudo mkdir -p /mnt/disks/gluster_data

# Deploying a cluster
cd cachew_experiments/experiments/autocaching

# Test that a cluster can be successfully deployed and terminated
./manage_cluster.sh start
./manage_cluster.sh stop