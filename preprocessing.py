import pandas as pd
import numpy as np
import os 
import requests

def preprocessing(f1='Taipower_20140101_20171231.csv', f2='TaiPower_20180101_20190131.csv', f3='TaiPower_20190101_20201031.csv'):
    ''' Load files '''
    path_14_17 = os.path.join('datas', f1)
    path_18_190131 = os.path.join('datas', f2)
    path_19_201031 = os.path.join('datas', f3)
    data_14_17 = pd.read_csv(path_14_17)
    data_18_190131 = pd.read_csv(path_18_190131)
    data_19_201031 = pd.read_csv(path_19_201031)
    ''' Format file 2014-2017 '''
    data_14_17['Date'] = pd.to_datetime(data_14_17['日期'], format="%Y/%m/%d").dt.strftime("%Y%m%d")
    useless_col = ['備轉容量率(%)', '日期']
    data_14_17 = data_14_17.drop(useless_col, axis=1)
    rename_dic = {'備轉容量(MW)': 'Reserve'}
    data_14_17.rename(columns=rename_dic, inplace=True)
    ''' Format file 2018 & 2019 '''
    col = ['日期', '備轉容量(MW)']
    data_18 = data_18_190131[data_18_190131['日期'] <= 20181231]
    data_19 = data_19_201031[data_19_201031['日期'] <= 20191231]
    data_20 =data_19_201031[data_19_201031['日期']>=20191231]
    data_18 = data_18[col]
    data_19 = data_19[col]
    data_20 = data_20[col]
    rename_dict = {'日期':'Date', '備轉容量(MW)':'Reserve'}
    data_18.rename(columns=rename_dict, inplace=True)
    data_19.rename(columns=rename_dict, inplace=True)
    data_20.rename(columns=rename_dict, inplace=True)
    
    #Download and format the newest data
    url_link='http://data.taipower.com.tw/opendata/apply/file/d006002/本年度每日尖峰備轉容量率.csv'
    url_content=requests.get(url_link).content
    with open("datas/Taipower_20210101_now.csv",'wb') as file:
        file.write(url_content)
    data_21=pd.read_csv("datas/Taipower_20210101_now.csv").iloc[:,:-1]
    data_21.iloc[:,1]=data_21.iloc[:,1]*10
    rename_dict = {'日期':'Date', '備轉容量(萬瓩)':'Reserve'}
    data_21.rename(columns=rename_dict,inplace=True)
    for i in range(len(data_21)):
        data_21.iloc[i,0]=str(data_21.iloc[i,0]).replace("/","")
    
    ''' Concatenate datas '''
    all_data = pd.concat((data_14_17, data_18, data_19,data_20,data_21))
    return all_data

if __name__ == "__main__":
    data=preprocessing
    data.to_csv("Testing.csv",index=False)