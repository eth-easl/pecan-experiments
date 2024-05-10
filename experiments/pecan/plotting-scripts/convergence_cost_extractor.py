import re
from datetime import datetime
import argparse

def get_summary_times(log_file):
    with open(log_file, 'r') as file:
        log_data = file.read()

    end_print_pattern = r'(.*): .*\] End Printing'
    end_print_times = re.findall(end_print_pattern, log_data)

    result = []
    for t in end_print_times:
        end_timestamp = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f').timestamp()
        end_timestamp = int(end_timestamp)
        result.append(end_timestamp)

    return result

def get_initial_time(log_file):
    with open(log_file, 'r') as file:
        log_data = file.read()

    pattern = r"(.*): .*\] After adding entry"
    start_times = re.findall(pattern, log_data)
    start_timestamp = datetime.strptime(start_times[0], '%Y-%m-%d %H:%M:%S.%f').timestamp()
    start_timestamp = int(start_timestamp)

    return start_timestamp

def get_number_of_remote_workers(log_file):
    with open(log_file, 'r') as file:
        log_data = file.read()

    block_pattern = r'EASL \(Heartbeat\) - Enough measurements for scalability metrics ((?:.|\n)*?) End Printing'
    summary_blocks = re.findall(block_pattern, log_data)
    number_of_workers = []
    for summary in summary_blocks:
        number_of_workers.append(get_number_of_summary_workers(summary))

    return number_of_workers

def get_number_of_summary_workers(summary: str):
    pattern = r".*\] Worker Address: otmraz-nodes-"
    oto_mraz_nodes = re.findall(pattern, summary)
    return len(oto_mraz_nodes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Log File Parser')
    parser.add_argument('--file', '-f', type=str, default='paper_plots\sensitivity_study\sensitivity_study_100_1_2.log')
    args = parser.parse_args()
    times = [get_initial_time(args.file)]
    times.extend(get_summary_times(args.file))
    convergence_times = []

    for i in range(len(times) - 1):
        convergence_times.append(times[i + 1] - times[i])

    number_of_remote_workers = get_number_of_remote_workers(args.file)
    interval_costs = []
    TPU_COST = 4.96
    REMOTE_WORKER_COST = 0.42
    for time, number_of_workers in zip(convergence_times, number_of_remote_workers):
        interval_costs.append((time/3600) * (TPU_COST + number_of_workers * REMOTE_WORKER_COST))

    print(sum(interval_costs))
