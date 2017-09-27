#!/bin/bash
# 从训练日志中抽取指定信息

# 暂存开始时间
start=$(date +%s)

if [ $# -eq 0 ]
then
    # datasets=10
    echo "Usage : $0 [start_index of datasets] [end_index of datasets] [phrase] [summary]"
    echo "Example : $0 1 100 MCC mcc.csv"
    exit 1
fi
: > $4
echo "Summary file '$4' cleaned up."
# 循环进行数据集训练测试
for i in `seq $1 $2`
do
    # 提取信息追加到新文件
    echo -n "f$i," >> $4
    # -n 配合p选项打印替换结果
    `sed -n s/$3\ =\ //p ./train_logs/f$i-train-c2-l3-u512-e400-lr0001-d01.log 1>>$4`
    # `python3 snn-k-fold-run.py -c 2 -l 3 -u 512 -e 300 -d 0.1 --learningrate 1e-2 -k 10 --inputfile ../citrullination_standard/$i.csv 1>../train_logs/f$i-train-c2-l3-u512-e300-lr001-d01.log 2>&1`
    # echo "Datasets $i training finished at `date`"
done

echo "Data extraction was successfully completed."
# 暂存结束时间
end=$(date +%s)

echo "Runtime: $((end-start)) seconds"
