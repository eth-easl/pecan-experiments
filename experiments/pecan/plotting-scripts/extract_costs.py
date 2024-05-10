import re

from absl import app
from absl import flags

import numpy as np

model_parameters = {
  "resnet": {
    "batch_size": 1024,
    "steps_per_epoch": 1251, 
  },
  "retinanet": {
    "batch_size": 64,
    "steps_per_epoch": 1848, 
  }
}

accelerator_costs = {
  "v2": 4.96,
  "v3": 8.8,
}

worker_costs = {
  'n2-8': 0.42,
}

flags.DEFINE_string('path', None, 'The path to the log file.')
flags.DEFINE_enum('model', 'retinanet', ['retinanet', 'resnet'], 
  'The type of the generated plot.')
flags.DEFINE_enum('accelerator', 'v2', ['v2', 'v3'], 
  'The type of the accelerator (TPUv2 or TPUv3).')
flags.DEFINE_enum('worker', 'n2-8', ['n2-8'], 
  'The type of the accelerator (TPUv2 or TPUv3).')
flags.DEFINE_integer('tpu_count', 1, 'The number of training clients.')
flags.DEFINE_integer('worker_count', 0, 'The number of worker nodes.')
flags.DEFINE_boolean('print_header', False, 'Print a CSV style header.')

FLAGS = flags.FLAGS


def main(argv):
  if FLAGS.path is None:
    print("You need to supply the path ")

  parameters = model_parameters[FLAGS.model]
  accelerator_cost = accelerator_costs[FLAGS.accelerator]
  worker_cost = worker_costs[FLAGS.worker]

  # Get a stable set of sampled examples/second measurements
  with open(FLAGS.path, "r") as f:
    values = []
    for line in f:
      hit = re.search(r"\d*\.*\d+[ ]+examples/second", line)
      if hit is not None:
        values.append(float(hit.group().split()[0]))

  arr_list = np.asarray(values)
  filter_bound = arr_list.max() - arr_list.max() * 0.05
  arr_list = arr_list[arr_list > filter_bound]

  mean_stable_sampling_time = arr_list.mean()

  # Infer the epoch time and costs
  batches_per_second = mean_stable_sampling_time / parameters["batch_size"]
  epoch_seconds = parameters["steps_per_epoch"] / batches_per_second
  hour_fraction = epoch_seconds / 3600

  actual_worker_cost = FLAGS.worker_count * hour_fraction *  worker_cost
  actual_tpu_cost = FLAGS.tpu_count * hour_fraction * accelerator_cost

  if FLAGS.print_header:
    print(f"tpu_cost,worker_cost,epoch_time,batches_sec")
  # Print the results: tpu_cost,worker_cost,epoch_time,batches_sec
  print(f"{actual_tpu_cost},{actual_worker_cost},{epoch_seconds},{batches_per_second}") 


if __name__ == '__main__':
  app.run(main)
