import pandas as pd
import numpy as np
import os 

def preprocessing(f1, f2, f3):
    ''' Load files '''
    path_14_17 = os.path.join('datas', f1)
    path_18_190131 = os.path.join('datas', f2)
    path_19_201031 = os.path.join('datas', f3)
    data_14_17 = pd.read_csv(path_14_17)
    data_18_190131 = pd.read_csv(path_18_190131)
    data_19_201031 = pd.read_csv(path_19_201031)
    ''' Format file 2014-2017 '''
    data_14_17['date'] = pd.to_datetime(data_14_17['日期'], format="%Y/%m/%d").dt.strftime("%Y%m%d")
    useless_col = ['備轉容量率(%)', '日期']
    data_14_17 = data_14_17.drop(useless_col, axis=1)
    rename_dic = {'備轉容量(MW)': 'reserve'}
    data_14_17.rename(columns=rename_dic, inplace=True)
    ''' Format file 2018 & 2019 '''
    col = ['日期', '備轉容量(MW)']
    data_18 = data_18_190131[data_18_190131['日期'] <= 20181231]
    data_19 = data_19_201031[data_19_201031['日期'] <= 20191231]
    data_18 = data_18[col]
    data_19 = data_19[col]
    rename_dict = {'日期':'date', '備轉容量(MW)':'reserve'}
    data_18.rename(columns=rename_dict, inplace=True)
    data_19.rename(columns=rename_dict, inplace=True)
    ''' Concatenate datas '''
    all_data = pd.concat((data_14_17, data_18, data_19))
    return all_data

if __name__ == "__main__":
    print(preprocessing('Taipower_20140101_20171231.csv', \
                        'TaiPower_20180101_20190131.csv', \
                        'TaiPower_20190101_20201031.csv'))