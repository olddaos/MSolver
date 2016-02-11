#!/bin/bash

while true
do
	date +"%Y-%m-%d %H:%M:%S,%3N"
	top -b -n1 | grep 'd.kor' | grep 'mpirun\|python'
	sleep 3
done

