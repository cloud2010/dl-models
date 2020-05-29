#!/bin/bash
# 完成所有方法在数据集上的测试任务
# 2020-5-29

if [ $# -eq 0 ]; then
    # datasets=10
    echo "Usage : $0 [datasets] [feature_selection_method]"
    echo "Example : $0 train_datasets.csv f_classif[chi2, mutual_info_classif]"
    exit 1
fi

# 暂存开始时间
start=$(date +%s)

# 激活python环境
# conda activate dl

for method in xgb lgb rf ext gdbt svm; do
    python clf_fs.py --datapath $1 -c $method -f $2
    echo "Datasets $1 was trained using the $method algorithm at $(date)"
done

# 暂存结束时间
end=$(date +%s)
echo "Runtime: $((end - start)) seconds"
