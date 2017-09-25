#!/bin/bash
# 完成所有数据集测试任务
# default setting

# 暂存开始时间
start=$(date +%s)

if [ $# -eq 0 ]
then
    # datasets=10
    echo "Usage : $0 [start_index of datasets] [end_index of datasets]"
    echo "Example : $0 1 100"
    exit 1
fi

# 循环进行数据集训练测试
for i in `seq $1 $2`
do
    # snn training
    # echo "$i.csv"
    `python3 snn-k-fold-run.py -c 2 -l 3 -u 512 -e 300 -d 0.1 --learningrate 1e-2 -k 10 --inputfile ../citrullination_standard/$i.csv 1>../train_logs/f$i-train-c2-l3-u512-e300-lr001-d01.log 2>&1`
    echo "Datasets $i training finished at `date`"
done

# 暂存结束时间
end=$(date +%s)

echo "Runtime: $((end-start)) seconds"