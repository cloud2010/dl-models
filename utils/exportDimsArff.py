"""
Export a custom dims feature matrix in ARFF format
"""
import time
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

__author__ = 'Min'


def pandas2arff(df, filename, wekaname="pandasdata", cleanstringdata=True, cleannan=True):
    """
    converts the pandas dataframe to a weka compatible file
    df: dataframe in pandas format
    filename: the filename you want the weka compatible file to be in
    wekaname: the name you want to give to the weka dataset (this will be visible to you when you open it in Weka)
    cleanstringdata: clean up data which may have spaces and replace with "_", special characters etc which seem to annoy Weka. 
                     To suppress this, set this to False
    cleannan: replaces all nan values with "?" which is Weka's standard for missing values. 
              To suppress this, set this to False
    """
    import re

    def cleanstring(s):
        if s != "?":
            return re.sub('[^A-Za-z0-9]+', "_", str(s))
        else:
            return "?"

    dfcopy = df  # all cleaning operations get done on this copy

    if cleannan != False:
        dfcopy = dfcopy.fillna(-999999999)  # this is so that we can swap this out for "?"
        # this makes sure that certain numerical columns with missing values don't get stuck with "object" type

    f = open(filename, "w")
    arffList = []
    arffList.append("@relation " + wekaname + "\n")
    # look at each column's dtype. If it's an "object", make it "nominal" under Weka for now (can be changed in source for dates.. etc)
    for i in range(df.shape[1]):
        if dfcopy.dtypes[i] == 'O' or (df.columns[i] in ["Class", "CLASS", "class"]):
            if cleannan != False:
                dfcopy.iloc[:, i] = dfcopy.iloc[:, i].replace(to_replace=-999999999, value="?")
            if cleanstringdata != False:
                dfcopy.iloc[:, i] = dfcopy.iloc[:, i].apply(cleanstring)
            _uniqueNominalVals = [str(_i) for _i in np.unique(dfcopy.iloc[:, i])]
            _uniqueNominalVals = ",".join(_uniqueNominalVals)
            _uniqueNominalVals = _uniqueNominalVals.replace("[", "")
            _uniqueNominalVals = _uniqueNominalVals.replace("]", "")
            _uniqueValuesString = "{" + _uniqueNominalVals + "}"
            arffList.append("@attribute " + df.columns[i] + _uniqueValuesString + "\n")
        else:
            arffList.append("@attribute " + df.columns[i] + " real\n")
            # even if it is an integer, let's just deal with it as a real number for now
    arffList.append("@data\n")
    for i in range(dfcopy.shape[0]):  # instances
        _instanceString = ""
        for j in range(df.shape[1]):  # features
            if dfcopy.dtypes[j] == 'O':
                _instanceString += "\"" + str(dfcopy.iloc[i, j]) + "\""
            else:
                _instanceString += str(dfcopy.iloc[i, j])
            if j != dfcopy.shape[1]-1:  # if it's not the last feature, add a comma
                _instanceString += ","
        _instanceString += "\n"
        if cleannan != False:
            _instanceString = _instanceString.replace(
                "-999999999.0", "?")  # for numeric missing values
            _instanceString = _instanceString.replace(
                "\"?\"", "?")  # for categorical missing values
        arffList.append(_instanceString)
    f.writelines(arffList)
    f.close()
    del dfcopy
    return True


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="This program is used to export a custom dims feature matrix in ARFF format.",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datapath", type=str, help="The path of dataset.", required=True)
    parser.add_argument("-i", "--idxpath", type=str,
                        help="The path of indexes file.", required=True)
    parser.add_argument("-b", "--start", type=int,
                        help="An integer number specifying at which position to start. Default is 0", required=True, default=0)
    parser.add_argument("-s", "--step", type=int,
                        help="An integer number specifying the incrementation. Default is 1", default=1)
    parser.add_argument("-e", "--stop", type=int,
                        help="An integer number specifying at which position to endt.", default=10)
    args = parser.parse_args()

    df_features = pd.read_csv(args.datapath, encoding='utf8')  # 读取整体特征矩阵
    df_features.iloc[:, 0] = df_features.iloc[:, 0].apply(
        lambda x: 'P'+str(x))  # 修改class列 (第1列) 显示值
    df_idxs = pd.read_table(args.idxpath, header=None)  # 读取特征索引文件

    print("\nThe shape of features matrix: ", df_features.shape)
    print("\nThe shape of indexes file: ", df_idxs.shape)
    print("\nStart:{0}, Stop:{1}, Step:{2}".format(args.start, args.stop, args.step))
    print("\nStart exporting custom features matrix...")

    # 根据步长大小等参数迭代输出不同维度特征矩阵
    for x in range(args.start, args.stop+1, args.step):
        if x == 0:
            pass
        else:
            new_features = df_features.iloc[:, np.append(df_idxs.loc[:(x-1)].values.T[0], 0)]
            pandas2arff(new_features, '{0}.arff'.format(x), wekaname='{0}_dims_features'.format(x))
            # new_features.to_csv('{0}.csv'.format(x), index=None)
            # print(df_features.iloc[:, np.insert(df_idxs.loc[:(x-1)].values.T[0], 0, 0)])

    end_time = time.time()  # 程序结束时间
    print("\n[Finished in: {0:.6f} mins = {1:.6f} seconds]".format(
        ((end_time - start_time) / 60), (end_time - start_time)))
