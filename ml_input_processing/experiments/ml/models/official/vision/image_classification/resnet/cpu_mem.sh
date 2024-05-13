#!/usr/bin/env bash

# Record CPU & memory usage over time
cpu_file='cpu.log'
mem_file='mem.log'

start=$(date +%s)
mem_total=$(cat /proc/meminfo | awk '/MemTotal/' | awk '{print $2}')

rm $cpu_file $mem_file

#while 'true'; do
#    echo $(($(date +%s) - $start)) $(expr 100 - $(vmstat 1 2 | awk '{print $15}' | tail -1)) >> $cpu_file
#    echo $(($(date +%s) - $start)) $(expr $mem_total - $(vmstat 1 2 | awk '{print $4}' | tail -1)) >> $mem_file
#    sleep 1
#done


while 'true'; do
    now=$(($(date +%s%N)/1000000))
    cpu=$(top -bn 2 -d 0.001 | grep '^%Cpu' | tail -n 1 | gawk '{print $2+$4+$6}')
    echo $(($now - $start)) $cpu >> $cpu_file
    #echo $(($(date +%s) - $start)) $(expr 100 - $(vmstat 1 2 | awk '{print $15}' | tail -1)) >> $cpu_file
    #echo $(($(date +%s) - $start)) $(expr $mem_total - $(vmstat 1 2 | awk '{print $4}' | tail -1)) >> $mem_file
    #sleep 1
done&

while 'true'; do
    nowmem=$(($(date +%s%N)/1000000))
    echo $(($nowmem - $start)) $(expr $mem_total - $(vmstat 1 2 | awk '{print $4}' | tail -1)) >> $mem_file
    #sleep 1
done&

