    # Set up the python environment
    echo "alias python=python3" >> .bashrc
    source .bashrc

    sudo apt update
    sudo apt install python3.8-venv -y
    python3 -m venv atc_venv
    source atc_venv/bin/activate

    # Set up the python environment and set up Python dependencies
    python3 -m pip install wheel
    gsutil cp gs://easl-atc24-ae-files/tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl .
    python3 -m pip install ./tensorflow-2.8.0-cp38-cp38-linux_x86_64.whl

    # Clone the relevant repository
    git clone -b main https://github.com/eth-easl/pecan-experiments.git
    python3 -m pip install -r pecan-experiments/requirements.txt

    ## Install other dependencies

    # Fix GCP TPU-VM bugs
    curl -O https://packages.cloud.google.com/apt/doc/apt-key.gpg 
    sudo apt-key add apt-key.gpg || sudo apt-key add apt-key.gpg
    (sudo apt update -y && sudo apt-get update) || (sudo apt update -y && sudo apt-get update)

    # kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    kubectl version --client

    # kops
    curl -Lo kops https://github.com/kubernetes/kops/releases/download/v1.19.3/kops-linux-amd64
    chmod +x ./kops
    sudo mv ./kops /usr/local/bin/

    # jq
    sudo apt-get install -y jq

    # GlusterFS - This section still causes trouble
    sudo add-apt-repository -y ppa:gluster/glusterfs-9
    sudo apt update -y
    sudo apt install -y glusterfs-client
    sudo mkdir -p /mnt/disks/gluster_data
