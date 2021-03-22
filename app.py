import os
import argparse
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression
import preprocessing
import fb_Prophet

if __name__ == '__main__':
	'''
	arguments to specify input/output files
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('--training',
						nargs = 2,
					   	default = ['training_data.csv', '本年度每日尖峰備轉容量率.csv'],
					   	help = 'input training data file name')
	parser.add_argument('--output',
						default = 'submission.csv',
						help = 'output file name')
	args = parser.parse_args()
	'''
	load datas from input file
	'''
	train_path = os.path.join('datas', args.training[0])
	#df_training = pd.read_csv(train_path)
	predict_path = os.path.join('datas', args.training[1])
	#df_predict = pd.read_csv(predict_path)
	'''
	Start prediction
	'''
	data=preprocessing.preprocessing()
	result=fb_Prophet.fitting_model(data,holiday_included=True,four_season=True)
	'''
	output result
	'''
	date = pd.to_datetime(result['ds'][-7:], format="%Y%m%d")
	result = result['yhat'][-7:]
	df_result = pd.DataFrame()
	df_result['date'] = date
	df_result['operating_reserve(MW)'] = result
	df_result.to_csv(args.output, index=0)
