#!/bin/bash
# sort 命令取 CSV 最大值

# 暂存开始时间
start=`date +"%s"`

if [ $# -eq 0 ]
then
    # datasets=10
    echo "Usage : $0 [file path]"
    echo "Example : $0 ~/rf-mcc.csv"
    exit 1
fi
echo -e "\nThe max:\n"
sort -t, -n -k2 $1 | tail -1 # -t 参数为分隔符 -k2 第2列

# 暂存结束时间
end=`date +"%s"`
ns=$((end-start))
echo -e "\n[Runtime: $ns seconds]\n"
