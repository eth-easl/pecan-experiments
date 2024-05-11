preprocessing_file="imagenet_preprocessing.py"
export PYTHONPATH=$HOME/ml_input_processing/experiments/ml/models/
export TF_DUMP_GRAPH_PREFIX=$HOME/ml_input_processing/experiments/ml/models/official/vision/image_classification/resnet/graph_dump.log
tpu_name="local"
model_dir="gs://otmraz-eu-logs/Resnet/ImageNet/${USER}"
data_dir="gs://tfdata-imagenet-eu" # This scripts needs 2 subfolders: train, validation
train_epochs=90