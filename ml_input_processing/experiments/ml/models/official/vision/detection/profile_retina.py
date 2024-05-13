from datetime import datetime
from packaging import version
import functools
from absl import app
from absl import flags
from absl import logging

import os, logging, time, math, sys

import tensorflow as tf
import random
import tensorflow_addons as tfa

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

FLAGS = flags.FLAGS
FLAGS(sys.argv)

from official.common import distribute_utils
from official.modeling.hyperparams import params_dict
from official.utils import hyperparams_flags
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
from official.vision.detection.configs import factory as config_factory

from official.vision.detection.dataloader import factory
from official.vision.detection.dataloader import mode_keys as ModeKeys
from official.vision.detection.dataloader import tf_example_decoder
from official.vision.detection.dataloader import input_reader
from official.vision.detection.utils import input_utils
from official.vision.detection.executor import distributed_executor as executor
from official.vision.detection.executor.detection_executor import DetectionDistributedExecutor
from official.vision.detection.modeling import factory as model_factory

hyperparams_flags.initialize_common_flags()
flags_core.define_log_steps()

flags.DEFINE_bool('enable_xla', default=False, help='Enable XLA for GPU')

flags.DEFINE_string(
    'mode',
    default='train',
    help='Mode to run: `train`, `eval` or `eval_once`.')

flags.DEFINE_string(
    'model', default='retinanet',
    help='Model to run: `retinanet`, `mask_rcnn` or `shapemask`.')

flags.DEFINE_string('training_file_pattern', None,
                    'Location of the train data.')

flags.DEFINE_string('eval_file_pattern', None, 'Location of ther eval data')

flags.DEFINE_string(
    'checkpoint_path', None,
    'The checkpoint path to eval. Only used in eval_once mode.')

flags.DEFINE_string(
    'dispatcher_ip', "otmraz-cachew-dispatcher-bl9v", 'Dispatcher IP')

flags.DEFINE_integer(
    "local_workers", 0, "number of local workers")

flags.DEFINE_bool('no_model', False, 'whether to run just the pipeline')

FLAGS = flags.FLAGS

DATA_AUGM_REPEAT = None
#CACHE_DIR = "/training-data/cache_temp"
CACHE_DIR = f"{os.getenv('HOME')}/training-data/cache_temp"
CACHE_PARALLELISM = 16
#DISPATCHER_IP='otmraz-cachew-dispatcher-qgvd'
DISPATCHER_IP=None
MAX_PIPELINING=8
MAX_OUTSTANDING_REQUESTS=96
TAKE1_CACHE_REPEAT=False

shuffle_buffer = 1000

eg_decoder = tf_example_decoder.TfExampleDecoder(include_mask=False)
_is_training = True

def _convert_to_target_type(image, data):
  image = input_utils.normalize_image(image)
  return image, data

def _decode(data):
  d = eg_decoder.decode(data)

  img = d.pop('image', None) # Remove the image key  
  return img, d

def _rand_flip(image, data):
    #image, data['groundtruth_boxes'] = input_utils.random_horizontal_flip(image, data['groundtruth_boxes'])

    if _is_training:
      classes = data['groundtruth_classes']
      boxes = data['groundtruth_boxes']
      is_crowds = data['groundtruth_is_crowd']
      # Skips annotations with `is_crowd` = True.
      if _is_training: #self._skip_crowd_during_training and self._is_training:
        num_groundtrtuhs = tf.shape(input=classes)[0]
        with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
          indices = tf.cond(
            pred=tf.greater(tf.size(input=is_crowds), 0),
            true_fn=lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
            false_fn=lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
        classes = tf.gather(classes, indices)
        boxes = tf.gather(boxes, indices)

      data['groundtruth_classes']=classes
      data['groundtruth_boxes']=boxes

    return image, data






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

DATA_DIR="gs://tfdata-datasets-eu/coco"
TRAIN_FILE_PATTERN=DATA_DIR+"/train-*"
EVAL_FILE_PATTERN=DATA_DIR+"/val-*"

training_file_pattern = FLAGS.training_file_pattern or TRAIN_FILE_PATTERN
files_list = tf.io.gfile.glob(training_file_pattern) #glob.glob(self._file_pattern)
print(files_list)

params = config_factory.config_generator(FLAGS.model)

params = params_dict.override_params_dict(
    params, FLAGS.config_file, is_strict=True)

params = params_dict.override_params_dict(
    params, FLAGS.params_override, is_strict=True)
params.override(
    {
        'strategy_type': FLAGS.strategy_type,
        'model_dir': FLAGS.model_dir,
        'strategy_config': executor.strategy_flags_dict(),
    },
    is_strict=False)
params.use_tpu = (params.strategy_type == 'tpu')

if not params.use_tpu:
    params.override({
        'architecture': {
            'use_bfloat16': False,
        },
        'norm_activation': {
            'use_sync_bn': False,
        },
    }, is_strict=True)

params.validate()
params.lock()
batch_size=params.train.batch_size
_parser_fn = factory.parser_generator(params, input_reader.ModeKeys.TRAIN)

dataset = tf.data.Dataset.from_tensor_slices(files_list)

_dataset_fn = tf.data.TFRecordDataset
dataset = dataset.interleave(
        map_func=_dataset_fn,
        cycle_length=32,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.apply(tf.data.experimental.mark("source_cache"))

dataset = dataset.shuffle(shuffle_buffer)

dataset = dataset.map(_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(lambda x, y: (input_utils.normalize_image(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.map(_rand_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE, keep_position=False)
dataset = dataset.map(_parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Use bfloat16 at the end
use_bfloat16 = True
dataset = dataset.map(lambda x, y: (tf.cast(x, dtype=tf.bfloat16), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.batch(batch_size, drop_remainder=True)

if DISPATCHER_IP is not None: # and DISPATCHER_IP !='loc':
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #if DISPATCHER_IP == 'loc':
    #  with open('disp_address.txt', 'r') as file:
    #    DISPATCHER_IP = file.read().replace('\n', '')
    dataset = dataset.apply(tf.data.experimental.service.distribute(
      processing_mode="distributed_epoch", service="grpc://" + DISPATCHER_IP + ":31000",
      max_outstanding_requests=32, max_request_pipelining_per_worker=2, compression=None, scaling_threshold_up=0.0,
        job_name="retina"
))

# Create a TensorBoard logging dir
#logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#print(logs)

#tf.profiler.experimental.start(logs)

batches = 0

starts = []
ends = []
durations = []
throughputs = []

for i in range(epochs):
    starts.append(time.time())
    print(i)
    for _ in dataset:
        if (batches % 100) == 0:
            print(batches)
            print(int(time.time()))
        batches += 1
        
#    for _ in ds_train:
        pass
    batches = 0
    ends.append(time.time())

#end = time.time()


#tf.profiler.experimental.stop()

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
