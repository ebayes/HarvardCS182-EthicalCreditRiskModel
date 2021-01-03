import pandas as pd
import numpy as np
import sys

#split sequence between HIGH risk and MEDIUM risk
def cut_sequence(ad_seq, at_seq, total_misses, x_len, y_len, customer_id):
	# We need to split input data into cases that were Medium Attention vs. High Attention
	ids_M = []
	quarter_M = []
	xs_M = []
	ys_M = []

	ids_H = []
	quarter_H = []
	xs_H = []
	ys_H = []

	misses_M = []
	misses_H = [] 

	#iterate through quarters
	for i in range(0, len(ad_seq)-x_len-y_len, 3):

		quarter = i/3
		ad_x = ad_seq[i:i+x_len]
		at_x = at_seq[i:i+x_len]
		at_y = at_seq[i+x_len:i+x_len+y_len]
		ad_y = ad_seq[i+x_len:i+x_len+y_len]

		if (at_x[-1] == 'M'):
			y=''
			# Then attention changed from med to high
			# We will define a change from med->high or high->med as 1 else 0
			if 'H' in at_y:
				y='1'
			else:
				y='0'

			ids_M.append(customer_id)
			quarter_M.append(quarter)
			misses_M.append(total_misses[i:i+x_len]) 

			arr_x = []
			for char in ad_x:
				arr_x.append(int(char))

			arr_y = []
			ad_y = ad_y + y
			for char in ad_y:
				arr_y.append(int(char))

			if len(arr_y) != 4:
				arr_y.append(int(y))

			xs_M.append(arr_x)
			ys_M.append(arr_y)

		# this is a High attention input
		elif (at_x[-1] == 'H'):
			y=''
			# Then attention changed from High to Mediums
			if 'M' in at_y:
				y='1'
			else:
				y='0'	
	
			ids_H.append(customer_id)
			quarter_H.append(quarter)
			misses_H.append(total_misses[i:i+x_len]) 
			
			arr_x = []
			for char in ad_x:
				arr_x.append(int(char))

			arr_y = []
			ad_y = ad_y + y

			for char in ad_y:
				arr_y.append(int(char))

			if len(arr_y) != 4:
				arr_y.append(int(y))

			xs_H.append(arr_x)
			ys_H.append(arr_y)

	return ids_M, quarter_M, xs_M, ys_M, ids_H, quarter_H, xs_H, ys_H, misses_M, misses_H

#separates and tehn recombines data into usable format
def restructure_data(df, x_len, y_len):
	
	adherence_sequences = df['AdherenceSequence'].values
	attention_sequences = df['AttentionString'].values
	ids = df['SK_ID_CURR']

	#columns
	all_ids_M = []
	all_ids_l_M = []
	all_times_M = []
	all_x_M = []
	all_y_M = []
	all_misses_M = []


	all_ids_H = []
	all_ids_l_H = []
	all_times_H = []
	all_x_H = []
	all_y_H = []
	all_misses_H = []

	# pulls out most recent 6 months for both adherence and attention
	for ad_sequence, at_sequence, id_i in zip(adherence_sequences,attention_sequences, ids):
		ad_sequence = ad_sequence[len(ad_sequence)::-1]
		at_sequence = at_sequence[len(at_sequence)::-1]

		#create binary sequence from string
		alt_ad_sequence = [1 if x!='0' else 0 for x in ad_sequence]

		#recreate string as binary sequence
		ad_sequence = ""
		for elem in alt_ad_sequence:
			ad_sequence += str(elem)  

		#calculate cumulative misses
		total_misses = np.array(alt_ad_sequence)
		total_misses = np.cumsum(total_misses)

		id_list_M, time_M, x_list_M, y_list_M, id_list_H, time_H, x_list_H, y_list_H, misses_M, misses_H = cut_sequence(ad_sequence, at_sequence, total_misses, x_len, y_len, id_i)

		all_ids_M += id_list_M
		all_times_M += time_M
		all_x_M += x_list_M
		all_y_M += y_list_M
		all_misses_M += misses_M

		all_ids_H += id_list_H
		all_times_H += time_H
		all_x_H += x_list_H
		all_y_H += y_list_H
		all_misses_H += misses_H
	
	#reshape arrays and concatenate
	all_ids_M = np.array(all_ids_M).reshape(-1,1)
	all_times_M = np.array(all_times_M).reshape(-1,1)
	all_x_M = np.array(all_x_M)
	all_y_M = np.array(all_y_M)
	all_misses_M = np.array(all_misses_M) 

	x_M = np.concatenate([all_ids_M, all_times_M, all_x_M], axis=1)
	print(all_y_M.shape)

	all_ids_H = np.array(all_ids_H).reshape(-1,1)
	all_times_H = np.array(all_times_H).reshape(-1,1)
	all_x_H = np.array(all_x_H)
	all_y_H = np.array(all_y_H)
	all_misses_H = np.array(all_misses_H)  

	x_H = np.concatenate([all_ids_H, all_times_H, all_x_H], axis=1)

	#concatenate all columns together
	all_data_M = np.concatenate([x_M, all_misses_M, all_y_M], axis=1)
	all_data_H = np.concatenate([x_H, all_misses_H, all_y_H], axis=1)

	print('M samples:',all_data_M.shape)
	print('H samples:',all_data_H.shape)

	print("Saving...")
	print(all_data_M[1])

  # output
	cols = ['SK_ID_CURR', 'QUARTER'] + ['x'+str(i + 1) for i in range(3)] + ['mis'+str(i + 1) for i in range(3)] + ['y'+str(i + 1) for i in range(3)] + ['label']
	data_M_df = pd.DataFrame(all_data_M, columns=cols)
	data_M_df.to_csv('data/processed/seq2attention_data_x'+str(x_len)+'y'+str(y_len)+'_M.csv',index=False)

	data_H_df = pd.DataFrame(all_data_H, columns=cols)
	data_H_df.to_csv('data/processed/seq2attention_data_x'+str(x_len)+'y'+str(y_len)+'_H.csv',index=False)

#quarter length
X_LEN = 3
Y_LEN = 3

# read in data
df = pd.read_csv('data/intermediate_data/bureau_agg.csv')

#select relevant columns
training_info = df[['SK_ID_CURR','AdherenceSequence','AttentionString']]

restructure_data(training_info, X_LEN, Y_LEN)