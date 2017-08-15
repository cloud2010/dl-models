# -*- coding: utf-8 -*-
"""
代码解释：
a) sys.argv[1:]为要处理的参数列表，sys.argv[0]为脚本名，所以用sys.argv[1:]过滤掉脚本名。
b) "hi:o:": 当一个选项只是表示开关状态时，即后面不带附加参数时，在分析串中写入选项字符。当选项后面是带一个附加参数时，在分析串中写入选项字符同时后面加一个":"号。所以"hi:o:"就表示"h"是一个开关选项；"i:"和"o:"则表示后面应该带一个参数。
c) 调用getopt函数。函数返回两个列表：opts和args。opts为分析出的格式信息。args为不属于格式信息的剩余的命令行参数。opts是一个两元组的列表。每个元素为：(选项串,附加参数)。如果没有附加参数则为空串''。

getopt函数的第三个参数[, long_options]为可选的长选项参数，上面例子中的都为短选项(如-i -o)
长选项格式举例:
--version
--file=error.txt
让一个脚本同时支持短选项和长选项
getopt.getopt(sys.argv[1:], "hi:o:", ["version", "file="])
"""
import sys
import getopt


def snn_usage():
    """SNN程序使用说明
    """
    print("\nThis a SelfNormalizingNetworks implementation using TensorFlow v1.2.1 framework.")
    print(
        "\nUsage:%s [-c|--h1|--h2] [--help|--inputFile] args...." % sys.argv[0])
    print("\nExample:python %s -c 2 --h1=256 --h2=256 --inputFile=train.data" %
          sys.argv[0])
    print("\nIntroduction:")
    print("\n-c: The number of class, e.g 2 or n")
    print("\n--h1: The number of hidden layer 1 units")
    print("\n--h2: The number of hidden layer 2 units")
    print("\n--inputFile: The filename of training dataset")


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hc:", ["h1=", "h2=",
                                                         "help", "inputFile="])
        if len(opts) == 0:
            snn_usage()
            sys.exit(1)
            
        print("============ opts ==================")
        print(opts, len(opts))

        print("============ args ==================")
        print(args)

        # check all params
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                snn_usage()
                sys.exit(1)
            elif opt not in ("--inputFile"):
                print("\nPlease input training dataset filename!")
                sys.exit(1)
            else:
                print("%s  ==> %s" % (opt, arg))
                """执行主函数"""
    except getopt.GetoptError:
        snn_usage()
        sys.exit(1)

# opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["version", "file="])
# input_file = ""
# output_file = ""
# file_name = ""
# if len(sys.argv) == 1:
#     print("\nUsage: python test.py  -h -i -o")
#     print(opts)
#     sys.exit()
# # if opts['file'] == 0:
# #     print('\nPlease input train data filename!')
# #     sys.exit()
# for op, value in opts:
#     if op == "-i":
#         input_file = value
#         print(input_file)
#     elif op == "-o":
#         output_file = value
#         print(output_file)
#     elif op == "--file":
#         file_name = value
#         print(file_name)
#     elif op == "--version":
#         print("v1.0")
#     elif op == "-h":
#         print("\nUsage: python test.py  -h -i -o")
#         print(opts[1])
#         sys.exit()

# import sys
# print("script name: ", sys.argv[0])
# print("\nargs lenth:", len(sys.argv))
# for i in range(1, len(sys.argv)):
#     print("paras%s:" % i, sys.argv[i])
