import tensorflow as tf
from imagenet_preprocessing import input_fn

from tqdm import tqdm

data_dir = "gs://tfdata-imagenet"
# nb_train_imgs=1281167 # ImageNet
nb_train_imgs=128116
#data_dir = "/training-data/imagenet-tiny/tfrecords"; nb_train_imgs=100000 # Tiny
#data_dir = "/training-data/imagenet_test/tfrecords"; nb_train_imgs=4 # contains 4 images

epochs = 4
batch_size = 312*4
steps_per_ep = nb_train_imgs//batch_size + 1
filenames = None

def process_epoch(dataset, steps_per_ep):
    i=0
    for _ in tqdm(dataset):
        i=i+1
        print(f"  batch {i}/{steps_per_ep}", end='\r')
        if i >= steps_per_ep:
            break

ds = input_fn(
    is_training=True,
    data_dir=data_dir,
    batch_size=batch_size,
    dtype=tf.float16,
    datasets_num_private_threads=32,
    drop_remainder=False,
    filenames=filenames,
)

tf.profiler.experimental.server.start(6009)

for i in range(epochs):
    print(f"Epoch {i+1}/{epochs}")
    process_epoch(ds, steps_per_ep)
    print("")

# tf.profiler.experimental.client.trace(
#     'grpc://localhost:6009,grpc://localhost:40001,grpc://localhost:40002',
#     './logs/tb_log',
#     2000
# )

# options = tf.profiler.experimental.ProfilerOptions(delay_ms=3000)
# tf.profiler.experimental.client.trace(
#     'grpc://localhost:6009,grpc://localhost:40001,grpc://localhost:40002',
#     './logs/tb_log',
#     20000
# )

