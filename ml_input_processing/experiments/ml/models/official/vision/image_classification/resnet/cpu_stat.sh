#!/usr/bin/env bash

path=$1

rm $path
while 'true' ; do
    echo $[100-$(vmstat 1 2|tail -1|awk '{print $15}')],`date +%s` >> $path
    sleep 1
done

