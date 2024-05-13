import sys
import time

if __name__ == "__main__":
    # first argument log_dir
    print("sleeping")
    time.sleep(100)
    import tensorflow as tf

    log_dir = sys.argv[1]
    n_local_workers = int(sys.argv[2])
    local_workers = [
        f"grpc://localhost:{idx + 38000}" for idx in range(n_local_workers)
    ]
    local_workers.append("grpc://localhost:6009")
    remote_workers = sys.argv[3:]
    remote_workers = [f"grpc://{remote_worker}" for remote_worker in remote_workers]

    print(log_dir, local_workers, remote_workers)

    for i in range(10):
        print("Start Profiling Round: ", i)
        try:
            tf.profiler.experimental.client.trace(
                ",".join(local_workers + remote_workers),
                log_dir,
                3000
            )
        except:
            print("Round ", i, " Failed!!")
        print("Sleeping")
        time.sleep(10)
