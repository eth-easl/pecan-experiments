#!/usr/bin/env bash

path=$1
dispatcher_pod_name=$2

while 'true' ; do
    timestamp=$(date +%s)
    kubectl logs $dispatcher_pod_name > ${path}/${timestamp}.log
    sleep 100
done

