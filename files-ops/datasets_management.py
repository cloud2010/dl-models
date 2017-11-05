"""
We will show how to load Membrane data into numpy array.
"""
import os
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from PIL import Image
from sklearn.preprocessing import StandardScaler
__author__ = 'Min'


if __name__ == "__main__":
    parser = ArgumentParser(description="This example will show how to load file data into numpy array.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--inputfile", type=str,
                        help="The filename of training dataset.", required=True)
    args = parser.parse_args()
    print("\nThe file path:", args.inputfile)
    result = list()
    with open(args.inputfile, 'r', encoding='utf-8') as f:
        # 读取首行分类信息
        labels = f.readline().split()
        for line in f.readlines():
            line = line.strip()
            # 过滤首行分类信息和文件末尾空行
            if len(line) > 0 and not line.startswith('>'):
                # 逐行存入矩阵信息
                result.append(line.split())
    # python list convert to numpy array
    mat = np.array(result, dtype=np.float)
    

    # padding array with 0, number of values padded to the footer
    if mat.shape[0] > 2000:
        mat = mat[0:2000, ]
    else:
        new_mat = np.lib.pad(
            mat, ((0, 2000 - mat.shape[0]), (0, 0)), 'constant', constant_values=0)
    scale = StandardScaler()
    scale.fit(new_mat)
    print("\nScale Mean:", scale.mean_)
    print("\n", new_mat)
    new_mat = scale.transform(new_mat)
    print("\nAfter Standardization:\n", new_mat)
    print("\n", new_mat.shape)
    new_mat = new_mat[:, :, np.newaxis]
    print(new_mat.shape)
    
    # img = Image.fromarray((new_mat * 255).astype(np.uint8))
    # img.save('test.jpg')
    # img.show()
    # img.save("test.jpg")
    # print("\nMembrane name: {0}, Class: {1}".format(
    #     labels[0].lstrip('>'), labels[1]))
    # print("\nMatrix shape:", mat.shape)
    # print("\nFirst line:", mat[1])

# 矩阵 padding 测试，矩阵尾部以 0 代替
# a = np.arange(9).reshape((3, 3))
# b = np.lib.pad(a, ((0, 5000 - 3), (0, 0)), 'constant', constant_values=0)
# print("\nBefore padding:", a.shape)
# print("\n", a)
# print("\nAfter padding:", b.shape)
# print("\n", b)
# c = b[0:5, ].reshape(1, 5, 3, 1)
# print("\n", c)
# c = np.append(c, b[6:11, ].reshape(1, 5, 3, 1), axis=0)
# print("\n", c.shape)
# a = np.array([1])
# b = np.array([2])
# a = np.hstack((a, b))
# print(a)