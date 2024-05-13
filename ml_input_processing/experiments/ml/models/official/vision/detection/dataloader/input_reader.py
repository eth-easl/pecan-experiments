# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data loader and input processing."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf

import os
import glob
from typing import Text, Optional
from official.modeling.hyperparams import params_dict
from official.vision.detection.dataloader import factory
from official.vision.detection.dataloader import mode_keys as ModeKeys
from official.vision.detection.dataloader import tf_example_decoder
from official.vision.detection.utils import input_utils
from official.vision.detection.exp_utils import get_disp_ip

DATA_AUGM_REPEAT = None
#CACHE_DIR = "/training-data/cache_temp"
CACHE_DIR = f"{os.getenv('HOME')}/training-data/cache_temp"
CACHE_PARALLELISM = 16

#DISPATCHER_IP='otmraz-cachew-dispatcher-bl9v'
#DISPATCHER_IP=None
MAX_PIPELINING=8
MAX_OUTSTANDING_REQUESTS=96
TAKE1_CACHE_REPEAT=False
DISTR = 4

shuffle_buffer = 1000

d = dict(os.environ)
DISPATCHER_IP=d["DISPATCHER_IP"]
if DISPATCHER_IP=='None':
  DISPATCHER_IP=None
DISPATCHER_IP=get_disp_ip(DISPATCHER_IP)
USE_AUTOORDER=d["USE_AUTOORDER"]
if USE_AUTOORDER == 'True':
  USE_AUTOORDER=True
else:
  USE_AUTOORDER=False

eg_decoder = tf_example_decoder.TfExampleDecoder(include_mask=False)

def _convert_to_target_type(image, data):
  print(type(data))
  image = input_utils.normalize_image(image)
  #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  print(type(data))
  return image, data

def _decode(data):
  d = eg_decoder.decode(data)
  print(type(d))
  img = d.pop('image', None)
  #img = tf.image.convert_image_dtype(img, dtype=tf.float32)
  #img = input_utils.normalize_image(img)
  return img, d

'''def _rand_flip(image, data):
  image, data['groundtruth_boxes'] = input_utils.random_horizontal_flip(image, data['groundtruth_boxes'])

  if self._is_training:
    classes = data['groundtruth_classes']
    boxes = data['groundtruth_boxes']
    is_crowds = data['groundtruth_is_crowd']
    # Skips annotations with `is_crowd` = True.
    if self._skip_crowd_during_training and self._is_training:
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
+
  return image, data'''

class InputFn(object):
  """Input function that creates dataset from files."""

  def __init__(self,
               file_pattern: Text,
               params: params_dict.ParamsDict,
               mode: Text,
               batch_size: int,
               num_examples: Optional[int] = -1):
    """Initialize.

    Args:
      file_pattern: the file pattern for the data example (TFRecords).
      params: the parameter object for constructing example parser and model.
      mode: ModeKeys.TRAIN or ModeKeys.Eval
      batch_size: the data batch size.
      num_examples: If positive, only takes this number of examples and raise
        tf.errors.OutOfRangeError after that. If non-positive, it will be
        ignored.
    """
    assert file_pattern is not None
    assert mode is not None
    assert batch_size is not None
    self._file_pattern = file_pattern
    self._mode = mode
    self._is_training = (mode == ModeKeys.TRAIN)
    self._batch_size = batch_size
    self._num_examples = num_examples
    self._parser_fn = factory.parser_generator(params, mode)
    self._dataset_fn = tf.data.TFRecordDataset
    
    self._example_decoder = tf_example_decoder.TfExampleDecoder(
        include_mask=False)

    self._input_sharding = (not self._is_training)
    try:
      if self._is_training:
        self._input_sharding = params.train.input_sharding
      else:
        self._input_sharding = params.eval.input_sharding
    except AttributeError:
      pass

  def _rand_flip(self, image, data):
    #image, data['groundtruth_boxes'] = input_utils.random_horizontal_flip(image, data['groundtruth_boxes'])

    if self._is_training:
      classes = data['groundtruth_classes']
      boxes = data['groundtruth_boxes']
      is_crowds = data['groundtruth_is_crowd']
      # Skips annotations with `is_crowd` = True.
      if self._is_training: #self._skip_crowd_during_training and self._is_training:
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

  def __call__(self, ctx=None, batch_size: int = None):
    """Provides tf.data.Dataset object.

    Args:
      ctx: context object.
      batch_size: expected batch size input data.

    Returns:
      tf.data.Dataset object.
    """
    if not batch_size:
      batch_size = self._batch_size
    assert batch_size is not None
    files_list = tf.io.gfile.glob(self._file_pattern) #glob.glob(self._file_pattern)
    print(files_list)
    dataset = tf.data.Dataset.from_tensor_slices(files_list)
    #dataset = dataset.shuffle(len(files_list))
    #dataset = tf.data.Dataset.list_files(
    #    self._file_pattern, shuffle=self._is_training)

    if self._input_sharding and ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    #dataset = dataset.cache()

    if self._is_training and DATA_AUGM_REPEAT is None and DISPATCHER_IP is None:
      #dataset = dataset.repeat()
      pass

    dataset = dataset.interleave(
        map_func=self._dataset_fn,
        cycle_length=32,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.apply(tf.data.experimental.mark("source_cache"))
    
    if self._is_training:
      dataset = dataset.shuffle(shuffle_buffer)
    if self._num_examples > 0:
      print("Take executed with {self._num_examples} -------------------------------------------------------") 
      dataset = dataset.take(self._num_examples)

    # Decode the dataset
    dataset = dataset.map(_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Convert first to float32
    #dataset = dataset.map(_convert_to_target_type)
    #dataset = dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y))

    dataset = dataset.map(lambda x, y: (input_utils.normalize_image(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if self._mode=="train":
        dataset = dataset.map(self._rand_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE, keep_position=USE_AUTOORDER==False)
    #dataset = dataset.map(lambda x, y: (tf.image.convert_image_dtype(x, dtype=tf.float32), y))

    # Parses the fetched records to input tensors for model function.
    dataset = dataset.map(self._parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Use bfloat16 at the end
    use_bfloat16 = True
    dataset = dataset.map(lambda x, y: (tf.cast(x, dtype=tf.bfloat16), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if DISPATCHER_IP is not None and self._is_training and DISTR == 3:
        dataset = dataset.apply(tf.data.experimental.service.distribute(
          processing_mode="distributed_epoch", service="grpc://" + DISPATCHER_IP + ":31000",
          max_outstanding_requests=MAX_OUTSTANDING_REQUESTS, max_request_pipelining_per_worker=MAX_PIPELINING, compression=None, job_name="RETINA" , fast_flow_offloading=49
        ))

    dataset = dataset.batch(batch_size, drop_remainder=True)

    if DISPATCHER_IP is not None and self._is_training and DISTR == 35:
        dataset = dataset.apply(tf.data.experimental.service.distribute(
          processing_mode="distributed_epoch", service="grpc://" + DISPATCHER_IP + ":31000",
          max_outstanding_requests=MAX_OUTSTANDING_REQUESTS, max_request_pipelining_per_worker=MAX_PIPELINING, compression=None, job_name="RETINA", fast_flow_offloading=50
        ))

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    if TAKE1_CACHE_REPEAT:
      dataset = dataset.take(1).cache().repeat()
    #return dataset.element_spec
    
    if DISPATCHER_IP is not None and self._is_training and DISTR == 4:
        dataset = dataset.apply(tf.data.experimental.service.distribute(
          processing_mode="distributed_epoch", service="grpc://" + DISPATCHER_IP + ":31000",
          max_outstanding_requests=MAX_OUTSTANDING_REQUESTS, max_request_pipelining_per_worker=MAX_PIPELINING, compression=None, job_name="RETINA"
        ))
    # Should be repeated also without a Cachew service
    dataset = dataset.repeat()

    # Caching
    if self._is_training and DATA_AUGM_REPEAT is not None:
        snapshot_path = CACHE_DIR+"/snapshot_{:03d}"
        
        cache_format=2
        cache_compression=1
        parallelism=8
        
        # Periodic caching using tf.data.experimental.service_cache_put
        dataset = dataset.apply(tf.data.experimental.service_cache_put(CACHE_DIR,
                                                                       cache_format=cache_format,
                                                                       cache_compression=cache_compression,
                                                                       parallelism=CACHE_PARALLELISM))
        element_spec = dataset.element_spec
        #tf.print("ELEMENT_SPEC", element_spec)
        cached_dataset = tf.data.experimental.serviceCacheGetDataset(CACHE_DIR,
                                                                     cache_format=cache_format,
                                                                     cache_compression=cache_compression,
                                                                     parallelism=CACHE_PARALLELISM,
                                                                     element_spec=element_spec)
        #cached_dataset=cached_dataset.shuffle(buffer_size=shuffle_buffer)
        cached_dataset = cached_dataset.repeat(DATA_AUGM_REPEAT-1)
        
        dataset = dataset.concatenate(cached_dataset)
        dataset = dataset.repeat()

        # Standard periodic caching
        #for ep in range(3):
        #    d = dataset.apply(tf.data.experimental.snapshot(snapshot_path.format(ep), compression=None))
        #    d = d.repeat(DATA_AUGM_REPEAT)
        #    if ep == 0:
        #        dataset_snapshot = d
        #    else:
        #        dataset_snapshot = dataset_snapshot.concatenate(d)
#
        #dataset = dataset_snapshot.repeat()

        #Partial periodic caching
        #cache_ratio = 0.5
        #nb_batches_to_cache = int(cache_ratio * NUM_IMAGES['train'] / batch_size)
        #for i in range(3):
        #    cache_dataset = dataset.take(nb_batches_to_cache)    
        #    fresh_dataset = dataset.skip(nb_batches_to_cache)
        #    cache_dataset = cache_dataset.apply(tf.data.experimental.snapshot(snapshot_path.format(i), compression=None))
        #    
        #    #d = cache_dataset.concatenate(fresh_dataset)
        #    weights = [cache_ratio, 1-cache_ratio]
        #    d = tf.data.experimental.sample_from_datasets([cache_dataset, fresh_dataset], weights, seed=None)
        #    
        #    d = d.repeat(DATA_AUGM_REPEAT)
        #    if i == 0:
        #        dataset_snapshot = d
        #    else:
        #        dataset_snapshot = dataset_snapshot.concatenate(d)
        #
        #dataset = dataset_snapshot.repeat()

        # Adaptive periodic caching (hard-coded for 6-3-1 caching periods with boundaries at epochs [30, 60])
        #for snap_nb in range(30//6):
        #    d = dataset.apply(tf.data.experimental.snapshot(snapshot_path.format(snap_nb%3), compression=None))
        #    d = d.repeat(6)
        #    if snap_nb == 0:
        #        dataset_snapshot = d
        #    else:
        #        dataset_snapshot = dataset_snapshot.concatenate(d)
        #
        #for snap_nb in range(30//6, 30//6 + (60-30)//3):
        #    d = dataset.apply(tf.data.experimental.snapshot(snapshot_path.format(snap_nb%3), compression=None))
        #    d = d.repeat(3)
        #    dataset_snapshot = dataset_snapshot.concatenate(d)
        #
        #d = dataset.repeat()
        #dataset = dataset_snapshot.concatenate(d)
    
    return dataset

  def _convert_to_target_type(data):
    data['image'] = tf.image.convert_image_dtype(data['image'], dtype=tf.float32)

    return data
