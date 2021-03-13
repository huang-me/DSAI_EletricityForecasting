import os
import argparse
import pandas as pd

if __name__ == '__main__':
	# arguments to specify input/output files
	parser = argparse.ArgumentParser()
	parser.add_argument('--training',
					   default = 'training_data.csv',
					   help = 'input training data file name')
	parser.add_argument('--output',
						default = 'submission.csv',
						help = 'output file name')
	args = parser.parse_args()
	electric_data = args.training
	output_file = args.output
	
	# load datas from input file
	train_path = os.path.join('datas', electric_data)
	df_training = pd.read_csv(train_path)

	# get data lenght
	length = len(df_training)

	# copy electric from days before
	days_before = 5
	for i in range(1, days_before+1, 1):
		df_training['{}b'.format(i)] = df_training['尖峰負載(MW)'].shift(-1*i)
		df_training['{}b_reserve'.format(i)] = df_training['備轉容量(MW)'].shift(-1*i)

	# extract feature
	feature = df_training[['1b', '2b', '3b', '4b', '5b', '1b_reserve', '2b_reserve', \
								'3b_reserve', '4b_reserve', '5b_reserve']][:length-5]
	print(feature)

	# target
	y = df_training['備轉容量(MW)'][:length-5]
	print(y)

	#model = Model()
	#model.train(df_training)
	#df_result = model.predict(n_step=7)
	#df_result.to_csv(args.output, index=0)
