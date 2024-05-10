import re
from datetime import datetime
import argparse

def get_convergence_time(log_file):
    with open(log_file, 'r') as file:
        log_data = file.read()

    # start_ts_pattern = r"Starting timestamp \(chorno\):\s+(\d+)"
    start_ts_pattern = r"(.*): .*\] After adding entry"
    end_print_pattern = r'(.*): .*\] End Printing'
    start_times = re.findall(start_ts_pattern, log_data)
    end_print_time = re.findall(end_print_pattern, log_data)

    start_timestamp = datetime.strptime(start_times[0], '%Y-%m-%d %H:%M:%S.%f').timestamp()
    start_timestamp = int(start_timestamp)

    end_timestamp = datetime.strptime(end_print_time[-1], '%Y-%m-%d %H:%M:%S.%f').timestamp()
    end_timestamp = int(end_timestamp)
    return end_timestamp - start_timestamp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Log File Parser')
    parser.add_argument('--file', '-f', type=str, required=True)
    args = parser.parse_args()
    print(get_convergence_time(args.file))
