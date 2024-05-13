from datetime import datetime
from packaging import version

import os, logging, time, math

import tensorflow as tf
from official.vision.image_classification.resnet import imagenet_preprocessing
import random
import tensorflow_addons as tfa

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

DISPATCHER_IP='otmraz-cachew-dispatcher-tsg6'
DISPATCHER_IP=None

data_dir="gs://tfdata-imagenet-eu"
DEFAULT_IMAGE_SIZE=224

num_local_workers=0

workers = []

#for i in range(num_local_workers):

if DISPATCHER_IP != None:
  loc_workers = tf.data.experimental.service.spawn_loc_workers(workers=num_local_workers, dispatcher=DISPATCHER_IP+":31000")



NUM_CHANNELS = 3
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

epochs = 2

f_names = imagenet_preprocessing.get_filenames(True, data_dir)[:41] # For Dan's dataset size
dataset = tf.data.Dataset.from_tensor_slices(f_names)

#dataset = dataset.shard(input_context.num_input_pipelines,
#                            input_context.input_pipeline_id)

dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      cycle_length=10,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.map(imagenet_preprocessing._decode_and_crop, tf.data.experimental.AUTOTUNE)

rads = 5 * math.pi / 180
dataset = dataset.map(lambda image, label: (tfa.image.shear_x(tfa.image.rotate(tf.image.random_flip_left_right(image), angles=random.uniform(0, rads)), level=random.uniform(-0.1, 0.1), replace=0), label), tf.data.experimental.AUTOTUNE)

#dataset = dataset.map(lambda image, label: (tf.image.random_flip_left_right(image), label), tf.data.experimental.AUTOTUNE)

dataset = dataset.map(lambda image, label: (imagenet_preprocessing._resize_image(image, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), label), tf.data.experimental.AUTOTUNE, keep_position=True)

dataset = dataset.map(lambda image, label: (imagenet_preprocessing._mean_image_subtraction(image, CHANNEL_MEANS, NUM_CHANNELS), label), tf.data.experimental.AUTOTUNE) #, keep_position=True)
dataset = dataset.map(lambda image, label: (tf.cast(image, tf.dtypes.float16),
                                              tf.cast(tf.cast(tf.reshape(label, shape=[1]), dtype=tf.int32) - 1, dtype=tf.float32)), tf.data.experimental.AUTOTUNE) #, keep_position=True)

dataset = dataset.batch(1024, drop_remainder=False)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

if DISPATCHER_IP is not None: # and DISPATCHER_IP !='loc':
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #if DISPATCHER_IP == 'loc':
    #  with open('disp_address.txt', 'r') as file:
    #    DISPATCHER_IP = file.read().replace('\n', '')
    dataset = dataset.apply(tf.data.experimental.service.distribute(
      processing_mode="distributed_epoch", service="grpc://" + DISPATCHER_IP + ":31000",
      max_outstanding_requests=32, max_request_pipelining_per_worker=2, compression=None, scaling_threshold_up=0.0,
        job_name="resnet"
))



'''(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.batch(128)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)
print(model.summary())'''

# Create a TensorBoard logging dir
logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
print(logs)

#tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                 histogram_freq = 1,
#                                                 profile_batch = '500,520')

tf.profiler.experimental.start(logs)

batches = 0

#start = time.time()
starts = []
ends = []
durations = []
throughputs = []
#card = 0

for i in range(epochs):
    starts.append(time.time())
    print(i)
    for _ in dataset:
        if (batches % 100) == 0:
            logging.info(batches)
        batches += 1
#    for _ in ds_train:
        pass
    ends.append(time.time())

#end = time.time()

'''model.fit(ds_train,
          epochs=3,
          validation_data=ds_test
          #, callbacks = [tboard_callback]
          )'''

tf.profiler.experimental.stop()

#duration = end - start
#print("Duration: " + str(duration))
#print("Throughput: " + str(1251/duration))

for i in range(len(ends)):
    durations.append(ends[i]-starts[i])
    throughputs.append((batches/epochs)/durations[i])

print("Starts: ")
print(starts)
print("Ends: ")
print(ends)
print("Durations: ")
print(durations)
print("Cardinality: " + str(batches / epochs))
print("Throughputs: ")
print(throughputs)
#print("Batches: ")


print("done")

