import os
import sys
import string

log_file = 'resnet_test.log'

def get_val_accuracy():
    cur_epoch = 0
    with open(log_file) as file:
        lines = file.readlines()
        
        for l in lines:
            if 'current_epoch :' in l:
                cur_epoch = l.replace('current_epoch : ','').strip(string.ascii_letters)[:-1]
            if ''''test_accuracy':''' in l:
                acc_str = str(round(float(l.split('test_accuracy\': ')[1].split(',')[0]), 3))
                print("Epoch: " + str(cur_epoch) + " Accuracy: " + acc_str)

get_val_accuracy()
