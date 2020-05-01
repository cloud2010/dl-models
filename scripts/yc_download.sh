#!/bin/bash

# for i in $(cat ${1}); do
#     echo $i
#     curl $'http://yzxt.shmtu.edu.cn/mrreexammanage/mrreexamnamelist\u0021print.action?type=ksbh&zzlx=A4' \
#         -H 'Connection: keep-alive' \
#         -H 'Cache-Control: max-age=0' \
#         -H 'Upgrade-Insecure-Requests: 1' \
#         -H 'Origin: http://yzxt.shmtu.edu.cn' \
#         -H 'Content-Type: application/x-www-form-urlencoded' \
#         -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.7 Safari/537.36' \
#         -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
#         -H 'Referer: http://yzxt.shmtu.edu.cn/mrreexammanage/mrreexamnamelist.action' \
#         -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,ja;q=0.6,da;q=0.5' \
#         -H 'Cookie: UM_distinctid=171821900e19ee-09f7ec8e8c28e5-6e52782d-1fa400-171821900e27c0; Hm_lvt_2e552fbaf5e9c8ff5d407c2048cc91bb=1587833646,1588173240,1588245134,1588335560; JSESSIONID=0B679CBCE680EA28E859533C00B68E1E; Hm_lpvt_2e552fbaf5e9c8ff5d407c2048cc91bb=1588349540' \
#         --data-raw 'foundataCourseId=&foundataOrienId=&maxRows=300&ids='$i'' \
#         --compressed \
#         --output $i.pdf \
#         --insecure
# done

# 分割符为,
while IFS=, read -r name id pro pdf; do
    ## Do something with $name, $id $pro and $pdf
    # echo $name
    # echo $pdf
    # 批量下载信息表
    curl $'http://yzxt.shmtu.edu.cn/mrreexammanage/mrreexamnamelist\u0021print.action?type=ksbh&zzlx=A4' \
        -H 'Connection: keep-alive' \
        -H 'Cache-Control: max-age=0' \
        -H 'Upgrade-Insecure-Requests: 1' \
        -H 'Origin: http://yzxt.shmtu.edu.cn' \
        -H 'Content-Type: application/x-www-form-urlencoded' \
        -H 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.7 Safari/537.36' \
        -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
        -H 'Referer: http://yzxt.shmtu.edu.cn/mrreexammanage/mrreexamnamelist.action' \
        -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,zh-TW;q=0.7,ja;q=0.6,da;q=0.5' \
        -H 'Cookie: UM_distinctid=171821900e19ee-09f7ec8e8c28e5-6e52782d-1fa400-171821900e27c0; Hm_lvt_2e552fbaf5e9c8ff5d407c2048cc91bb=1587833646,1588173240,1588245134,1588335560; JSESSIONID=0B679CBCE680EA28E859533C00B68E1E; Hm_lpvt_2e552fbaf5e9c8ff5d407c2048cc91bb=1588349540' \
        --data-raw 'foundataCourseId=&foundataOrienId=&maxRows=300&ids='$pdf'' \
        --compressed \
        --output $id\_$name\_$pro.pdf \
        --insecure
done <${1}
