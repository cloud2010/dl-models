#!/bin/bash

# max_depth para
md=$1

# default setting
if [ $# -eq 0 ]
then
	echo "default test times is 3."
	md=3
	# echo "Usage : $0 number"
	# exit 1
fi

for ((i=1; i<=$md; i++)) 
do
	# gdbt caculate
	echo "md $i Process running...`python3.6 gdbt.py -cv 10 -md $i heart_scale 1>heart_scale-cv-10-md-$i.log 2>&1`"
done
