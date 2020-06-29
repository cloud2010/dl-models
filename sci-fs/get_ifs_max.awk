#! /usr/bin/gawk -f

BEGIN {
    FS=","; # 指定CSV分隔符
    print "\nF","\t","Max";
    print "-----------------";
    max=0;
}
# 过滤第一行 输出MCC列最大值
NR>1{
    if($5>max) {max=$5; f=$1}
}
END {
    print f,"\t",max;
    #print var; # 打印传入参数
}

