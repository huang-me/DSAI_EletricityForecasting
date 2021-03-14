import os
import argparse
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression

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
	df_training = pd.read_csv(train_path)
	predict_path = os.path.join('datas', args.training[1])
	df_predict = pd.read_csv(predict_path)
	'''
	get data lenght
	'''
	length = len(df_training)
	'''
	copy electric from days before
	'''
	days_before = 10
	for i in range(1, days_before+1, 1):
		df_training['{}b_reserve'.format(i)] = df_training['備轉容量(MW)'].shift(i)
	for i in range(0, 7, 1):
		df_training['{}_ans'.format(i+1)] = df_training['備轉容量(MW)'].shift(-1*i)
	'''
	extract feature
	'''
	feat_name = []
	for i in range(1, days_before+1):
		feat_name.append('{}b_reserve'.format(i))
	feature = df_training[feat_name][days_before:-7]
	# print(feature)
	'''
	target
	'''
	y = df_training[['1_ans', '2_ans', '3_ans', '4_ans', '5_ans', '6_ans', '7_ans']][days_before:-7]
	# print(y)
	'''
	fit model
	'''
	model = LinearRegression()
	model.fit(feature, y)
	'''
	format predict file
	'''
	df_predict['備轉容量(MW)'] = df_predict['備轉容量(萬瓩)'] * 10
	df_predict = df_predict.drop(['備轉容量(萬瓩)', '備轉容量率(%)'], axis=1)
	# print(df_predict)
	'''
	feature of predict
	'''
	predict_x = [df_predict['備轉容量(MW)'][-1*days_before:]]
	# print(predict_x)
	'''
	output result
	'''
	date = pd.to_datetime(df_predict['日期'][-7:], format="%Y/%m/%d")
	date = (date + timedelta(days=7)).dt.strftime("%Y%m%d")
	result = model.predict(predict_x).transpose()
	df_result = pd.DataFrame()
	df_result['Time'] = date
	df_result['result'] = result
	df_result.to_csv(args.output, index=0, header=0)
