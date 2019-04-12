#!/bin/bash
# 从训练日志中抽取指定信息
# 添加 for nested loop
# tsvm 

# 暂存开始时间
start=$(date +%s)

if [ $# -eq 0 ]
then
    # datasets=10
    echo "Usage : $0 [start_index of datasets] [end_index of datasets] [phrase] [summary]"
    echo "Example : $0 1 100 MCC mcc.csv"
    exit 1
fi
# : > $4
# echo "Summary file '$4' cleaned up."
# 循环进行数据集训练测试
for j in {1..100}
do
    for i in `seq $1 $2`
    do
        # 提取信息追加到新文件
        echo -n "f$i," >> $4
        # -n 配合p选项打印替换结果
        `sed -n s/$3\ =\ //p /home/train_logs/svm/seed-$j/f$i-train.log 1>>seed-$j-$4`
    done
    echo "seed $j: '$3' data extraction was successfully completed."
done

# 暂存结束时间
end=$(date +%s)

echo "Runtime: $((end-start)) seconds"
