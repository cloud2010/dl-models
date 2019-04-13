#!/bin/bash
# 从 Seed 1..100 中抽取每个训练日志的 MAX value
# Random Forest

# 暂存开始时间
start=$(date +%s)

if [ $# -lt 3 ]
then
    # datasets=10
    echo "Usage : $0 [start_index of seed] [end_index of seed] [mcc or acc] [logs dir path]"
    echo "Example : $0 1 100 mcc /home/train_logs/rf/csv"
    exit 1
fi
# : > $4
# echo "Summary file '$4' cleaned up."
# 循环进行数据集训练测试
for i in `seq $1 $2`
do
    # 提取最大值信息
    # echo -n "seed-$i," >> $4
    gawk -F ',' 'BEGIN { max=0 } $2 > max { max=$2;f=$1 } END { print "'"seed-$i"'",f, max }' $4/seed-$i-$3.csv
    # -n 配合p选项打印替换结果
    # `sed -n s/$3\ =\ //p /home/train_logs/rf/f$i-train.log 1>>$4`
done

echo "Max data extraction was successfully completed."

# 暂存结束时间
end=$(date +%s)

echo "Runtime: $((end-start)) seconds = $(((end-start)/60)) mins"
