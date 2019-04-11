#!/bin/bash
# 完成所有数据集测试任务 Random Forest
# 添加嵌套循环 1-100 random seed date: 2019-04-11
# 首先确保激活 venv
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
for j in {1..100}
do
    for i in `seq $1 $2`
    do
        # start training
        # echo "$i.csv"
        `python $HOME/workspaces/dl-models/rf-k-fold-run.py --datapath $HOME/citrullination_standard_941/$i.csv -r $j 1>$HOME/train_logs/rf/seed-$j/f$i-train.log 2>&1`
        echo "Seed $j: Datasets $i training finished at `date`"
    done
done

# 暂存结束时间
end=$(date +%s)

echo "Runtime: $((end-start)) seconds"
