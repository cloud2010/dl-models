#!/bin/bash
# 完成所有数据集测试任务
# default setting
if [ $# -eq 1 ]
then
	echo "default test datasets is 941."
	datasets=941
	# echo "Usage : $0 number"
	# exit 1
fi

for ((i=1; i<=$datasets; i++)) 
do
	# gdbt caculate
	# echo "md $i Process running...`python3.6 gdbt.py -cv 10 -md $i heart_scale 1>heart_scale-cv-10-md-$i.log 2>&1`"
done
