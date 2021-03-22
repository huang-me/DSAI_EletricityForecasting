from fbprophet.diagnostics import cross_validation
from fbprophet import Prophet
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric,plot_plotly, plot_components_plotly
from math import sqrt
from datetime import datetime

import os
from scipy import stats
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

dataset_path="Dataset/Weather/"

def remove_outlier(data,varname):
    remove_threshold=3
    #Remove outlier by using z-score
    
    z_score=np.abs(stats.zscore(data[varname]))
    outlier=np.array(np.where(z_score>remove_threshold))
    outlier=np.reshape(outlier,(outlier.shape[1]))

    '''
    if np.asarray(outlier).size>0:
        print(f"Outlier length:{data.iloc[outlier]}")
    else:
        print("No outlier")
    '''
    data=data.drop(data.index[outlier])
    return data

def getting_holiday_info(path):
    df=pd.read_csv(path)
    df=df[df["Holiday Type"]=="National holiday"]

    #Some holiday doesn't contribute anything to the trend but noise
    #These holiday can be removed
    holiday_removed_array=['Dragon']
    for holiday_removed in holiday_removed_array:
        df=df[~df["Holiday Name"].str.contains(holiday_removed)] 

    #new holiday date
    new_date=pd.to_datetime(df["Date"],format="%Y-%m-%d")
    

    return_data=pd.DataFrame({"ds":new_date,"holiday":df["Holiday Name"],"lower_window":0,"upper_window":0})
    return_data.to_csv("holiday.csv",index=False)
    return return_data

def get_weather_info(weather_csv,date):
    temperature=[]
    for i in date:
        i=str(i).split(' ')[0]
        temp=weather_csv[weather_csv['Date']==str(i)]["Temperature"]
        if temp.empty:
            print(str(i)+" doesn't have weather data")
            print("30 degree is give as fake data")
            temp=30
        else:
            temp=temp.item()
        temperature.append(temp)
    return temperature

def summer_wintter_spring_auttum(df):
    ds=pd.to_datetime(df["ds"],format="%Y-%m-%d")
    
    spring=[]
    summer=[]
    winter=[]
    auttum=[]
    for date in ds:
        month=date.month

        if month <4:
            spring.append(True)
            winter.append(False)
            summer.append(False)
            auttum.append(False)
        elif month<7:
            spring.append(False)
            winter.append(False)
            summer.append(True)
            auttum.append(False)
        elif month<10:
            spring.append(False)
            winter.append(False)
            summer.append(False)
            auttum.append(True)
        else:
            spring.append(False)
            winter.append(True)
            summer.append(False)
            auttum.append(False)

    return spring,summer,winter,auttum



    
def validation(model,data,city,y_predict,number_of_day_initial,period,horizon):
    '''
    period set how frequenly will the prediction be carried out
    horizon define long each forcast will be 
    forecast will be happening at cutoff+horizon
    '''
    df_cv=cross_validation(model,initial=number_of_day_initial,period=period,horizon=horizon)
    df_p=performance_metrics(df_cv)
    fig4=plot_cross_validation_metric(df_cv,metric='rmse')
    fig4.canvas.set_window_title("cross validation")

    #calculating the mean absolute percentage error(MAPE)
    mape=mean_absolute_percentage_error(df_cv.y,df_cv.yhat)
    print(f"The mape of train data is {mape}%")


def on_CNY_season(data):
    #This function will determine whether the date was a week before CNY

    df=pd.read_csv("datas/All_Holiday.csv")
    df=df[df["Holiday Name"]=="Chinese New Year's Eve"]
    df_date=pd.to_datetime(df["Date"],format="%Y-%m-%d")

    year=[]
    for i in df_date:
        year.append(i.year)
    df["year"]=year

    CNY_season=[]
    yearly_enable=[]
    for ds in data["ds"]:
        date_refer=pd.to_datetime(ds,format="%Y-%m-%d")
        year_of_date_refer=date_refer.year
        month_of_date_refer=date_refer.month

        if month_of_date_refer >2:
            CNY_season.append(False)
            yearly_enable.append(True)
            continue


        CNY_Date=pd.to_datetime(df[df["year"]==year_of_date_refer]["Date"])
        date_two_week_before_CNY=CNY_Date-pd.Timedelta(days=14)
        temp=(date_refer>date_two_week_before_CNY) & (date_refer<CNY_Date)
        CNY_season.append(temp.item())

        date_week_after_CNY=CNY_Date+pd.Timedelta(days=7)
        temp=~(date_refer>date_two_week_before_CNY) & (date_refer<date_week_after_CNY)
        yearly_enable.append(temp.item())
    

    return CNY_season,yearly_enable

def rmse(y_predicted, y_actual):
    return sqrt(mean_squared_error(y_actual, y_predicted))

def fitting_model(data,weather_included=False,holiday_included=False,CNY_season=False,last_7_days_validation=False,four_season=False):
    #change the format of date
    data_copy=data.copy()
    temp=[]
    for i in range (len(data_copy)):
        str_date=str(data_copy.iloc[i]["Date"])
        temp.append(datetime.strptime(str_date,'%Y%m%d'))
    data_copy['Date']=temp
    data_copy=remove_outlier(data_copy,'Reserve')
    #Config the model used
    #m=Prophet(daily_seasonality=False,changepoint_prior_scale=0.1,holidays_prior_scale=0.1)
    m=Prophet(daily_seasonality=False)
    if holiday_included:
        m.holidays=getting_holiday_info("datas/All_Holiday.csv")

    if CNY_season:
        m.weekly_seasonality=False
        m.yearly_seasonality=False
        m.add_seasonality(name="Weekly on CNY Season",period=7,fourier_order=3,condition_name="CNY season")
        m.add_seasonality(name="Weekly on other dates",period=7,fourier_order=3,condition_name="Other season")

    if four_season:
        m.weekly_seasonality=False
        m.yearly_seasonality=False
        m.add_seasonality(name="Spring Season",period=91.5,fourier_order=5,prior_scale=1,condition_name='Spring')
        m.add_seasonality(name="Summer Season",period=91.5,fourier_order=5,prior_scale=1,condition_name='Summer')
        m.add_seasonality(name="Autumn Season",period=91.5,fourier_order=5,prior_scale=1,condition_name='Autumn')
        m.add_seasonality(name="Winter Season",period=91.5,fourier_order=5,prior_scale=1,condition_name='Winter')

    

    data_history=pd.DataFrame({'ds':data_copy["Date"],"y":data_copy['Reserve']})

    if last_7_days_validation:
        data_history=data_history[:-7]


    if weather_included:
        #Get temperature info
        '''
        weather_csv=pd.read_csv(weather_csv_location[city])
        date=data["Date"]
        temperature=get_weather_info(weather_csv,date)
        data_history["Weather"]=temperature
        m.add_regressor('Weather')
        '''
    if CNY_season:
        data_history["CNY season"],data_history["Yearly season"]=on_CNY_season(data_history)
        data_history["Other season"]=~data_history["CNY season"]
    if four_season:
        data_history["Spring"],data_history["Summer"],data_history["Autumn"],data_history["Winter"]=summer_wintter_spring_auttum(data_history)

    #Predict one year ahead
    m.fit(data_history)
    future_date = m.make_future_dataframe(periods=7)

    if weather_included:
        '''
        temp=pd.to_datetime(future_date['ds'],format="%Y-%m-%d")
        temperature=get_weather_info(weather_csv,temp.dt.date)
        future_date["Weather"]=temperature
        '''

    if CNY_season:
        future_date["CNY season"],future_date["Yearly season"]=on_CNY_season(future_date)
        future_date["Other season"]=~future_date["CNY season"]

    if four_season:
        future_date["Spring"],future_date["Summer"],future_date["Autumn"],future_date["Winter"]=summer_wintter_spring_auttum(future_date)

    #predict future price
    
    future=m.predict(future_date)

    #fig3=m.plot(future)
    #fig3.canvas.set_window_title("Prediction")
    #fig2=m.plot_components(future)
    #fig2.canvas.set_window_title("Component")

    #plot the corss validation erro
    #validation(m,data,city,future,"1825 days","100 days","100 days")
    print(data_copy[-7:])
    if last_7_days_validation:
        print(f"RMSE:{rmse(future['yhat'].tail(7),data_copy[-7:].Reserve)}")
    return future


if __name__=='__main__':
    import preprocessing
    data=preprocessing.preprocessing()
    result=fitting_model(data,holiday_included=True,last_7_days_validation=True,CNY_season=False,four_season=True)
    print(result['ds'].tail(7))
    plt.show()
