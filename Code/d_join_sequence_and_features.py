import pandas as pd 
import numpy as np 
import os

#quarter length
x_len = 3
y_len = 3

sequence_M_df = pd.read_csv('data/processed/seq2attention_data_x'+str(x_len)+'y'+str(y_len)+'_M.csv')
sequence_H_df = pd.read_csv('data/processed/seq2attention_data_x'+str(x_len)+'y'+str(y_len)+'_H.csv')

#identify relevant static features
feature_columns = ['SK_ID_CURR', 'TARGET', 'EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1',
                  'DAYS_BIRTH', 'NAME_EDUCATION_TYPE_Higher education', 'CODE_GENDER',
                  'NAME_INCOME_TYPE_Pensioner', 'DAYS_EMPLOYED', 'ORGANIZATION_TYPE_XNA',
                  'FLOORSMAX_AVG', 'FLOORSMAX_MEDI', 'FLOORSMAX_MODE',
                  'EMERGENCYSTATE_MODE_No', 'HOUSETYPE_MODE_block of flats', 'AMT_GOODS_PRICE',
                  'REGION_POPULATION_RELATIVE', 'ELEVATORS_AVG', 'ELEVATORS_MEDI',
                  'FLOORSMIN_AVG', 'FLOORSMIN_MEDI', 'WALLSMATERIAL_MODE_Panel',
                  'LIVINGAREA_AVG', 'LIVINGAREA_MEDI', 'FLOORSMIN_MODE', 'TOTALAREA_MODE',
                  'ELEVATORS_MODE', 'NAME_CONTRACT_TYPE_Cash loans', 'LIVINGAREA_MODE', 'AMT_CREDIT']

# months in program with possible joinable entries
dfs_M = []
dfs_H = []

print("Looping")
for i in range(1, 31):

	feature_df = pd.read_csv('data/intermediate_data/application_train_test.csv')

	feature_df = feature_df[feature_columns]

	#filter by quarter and risk
	sequence_M_filtered_df = sequence_M_df[sequence_M_df['QUARTER']==i]
	sequence_H_filtered_df = sequence_H_df[sequence_H_df['QUARTER']==i]

	#merge
	merged_M_df = pd.merge(feature_df, sequence_M_filtered_df, how='inner', left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'])
	merged_H_df = pd.merge(feature_df, sequence_H_filtered_df, how='inner', left_on=['SK_ID_CURR'], right_on=['SK_ID_CURR'])

	dfs_M.append(merged_M_df)
	dfs_H.append(merged_H_df)

print("Formatting")
final_M_df = pd.concat(dfs_M,axis=0)
final_H_df = pd.concat(dfs_H,axis=0)

#check running correctly
print(final_M_df)
print(final_H_df)

print(final_M_df.head())
print(final_H_df.head())
print(final_M_df.shape)
print(final_H_df.shape)
print(final_M_df.columns)
print(final_H_df.columns)

final_M_df = final_M_df.fillna(0)
final_H_df = final_H_df.fillna(0)

final_M_df = final_M_df.sample(frac=1) # shuffling happens here
final_H_df = final_H_df.sample(frac=1) # shuffling happens here

final_M_df.to_csv('data/processed/train_and_test_x%iy%i_M.csv'%(x_len, y_len))
final_H_df.to_csv('data/processed/train_and_test_x%iy%i_H.csv'%(x_len, y_len))

