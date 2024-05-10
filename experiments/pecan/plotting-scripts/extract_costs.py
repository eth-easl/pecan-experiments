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
flags.DEFINE_integer('worker_matches', 3, 'The number of matches in worker configs '
                     'before declaring it a convergence')
flags.DEFINE_boolean('infer_workers', True, 'If True, infers worker count from logs.')
flags.DEFINE_boolean('header', False, 'Print a CSV style header.')

FLAGS = flags.FLAGS




def main(argv):
  if FLAGS.path is None:
    print("You need to supply the path ")

  parameters = model_parameters[FLAGS.model]
  accelerator_cost = accelerator_costs[FLAGS.accelerator]
  worker_cost = worker_costs[FLAGS.worker]

  assessing_deployment = False
  stable_remote_worker_count = 0
  stable_local_worker_count= 0

  # Get a stable set of sampled examples/second measurements
  with open(FLAGS.path, "r") as f:
    values = []
    for line in f:
      hit = re.search(r"\d*\.*\d+[ ]+examples/second", line)
      if hit is not None:
        values.append(float(hit.group().split()[0]))

  # Read the file backwards and try to find the converged number of workers
  if FLAGS.infer_workers:
    identical_count = 0
    with open(FLAGS.path, "r") as f:
      local_workers = set()
      remote_workers = set()
      for line in reversed(list(f)):
        if assessing_deployment and ("ClientHeartbeat: Normal Tasks" in line or "ClientHeartbeat: Current Tasks" in line):
          assessing_deployment = False
          observed_local_workers = len(local_workers)
          observed_remote_workers = len(remote_workers)
          if identical_count >= 1 and stable_remote_worker_count == observed_remote_workers\
            and stable_local_worker_count == observed_local_workers:
            identical_count += 1
            if identical_count >= FLAGS.worker_matches:
              break
          else:
            # Either different deployment or not initialized
            stable_remote_worker_count = observed_remote_workers
            stable_local_worker_count = observed_local_workers
            identical_count = 1
        elif "End Printing" in line:
          assessing_deployment = True
          local_workers.clear()
          remote_workers.clear()
        elif assessing_deployment:
          hit = re.search(r"Worker Address:\s[0-9\-a-zA-Z:]+", line)
          if hit is not None:
            worker_name = hit.group().split()[2]
            if "localhost" in worker_name:
              local_workers.add(worker_name)
            else:
              remote_workers.add(worker_name)

  arr_list = np.asarray(values)
  filter_bound = arr_list.max() - arr_list.max() * 0.05
  arr_list = arr_list[arr_list > filter_bound]

  mean_stable_sampling_time = arr_list.mean()

  # Infer the epoch time and costs
  batches_per_second = mean_stable_sampling_time / parameters["batch_size"]
  epoch_seconds = parameters["steps_per_epoch"] / batches_per_second
  hour_fraction = epoch_seconds / 3600

  actual_worker_cost = stable_remote_worker_count * hour_fraction *  worker_cost
  actual_tpu_cost = FLAGS.tpu_count * hour_fraction * accelerator_cost

  if FLAGS.header:
    print(f"tpu_cost,worker_cost,remote_workers,local_workers,epoch_time,batches_sec")
  print(f"{actual_tpu_cost},{actual_worker_cost},{stable_remote_worker_count},"
        f"{stable_local_worker_count},{epoch_seconds},{batches_per_second}") 


if __name__ == '__main__':
  app.run(main)
