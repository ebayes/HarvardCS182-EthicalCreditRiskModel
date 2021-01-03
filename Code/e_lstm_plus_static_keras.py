import pandas as pd
import numpy as np
import random
from imblearn.over_sampling import SMOTE
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
import models_keras as models
import sys
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer as myScaler


#helper functions
def threshold_predicitons(p, t):
	out = []
	for entry in p:
		if entry < t:
			out.append(0)
		else:
			out.append(1)
	return np.array(out)

def get_calls_of_correct_positives(cp, y):
	y_out = []
	for i in range(cp.shape[0]):
		if cp[i] == 1:
			y_out.append(y[i])
	return np.array(y_out)

def get_false_positives(preds, actuals):
	out = []
	for i in range(preds.shape[0]):
		if preds[i] == 1 and actuals[i] == 0:
			out.append(1)
	return np.array(out)

def print_total_missed_calls_from_high_risk(y_test, y_test_calls):
	a = []
	for i in range(y_test.shape[0]):
		if y_test[i] == 1:
			a.append(y_test_calls[i])
	a = np.array(a)
	a = abs(1 - a)
	print("Total missed calls to find:",a.sum())

def restructure_data(df, split=0.25):

	df = df[cols_id+cols_static+cols_sequence_x+cols_sequence_y+cols_label]

	data = df.values

	split_index = int(data.shape[0]*(1-split))
	train, test = data[:split_index], data[split_index:]
	print(train[0])

	id_train = train[:,0]
	week_train = train[:,1]
	x_static_train = train[:, 2:2+len(cols_static)].astype(float)
	x_sequence_train = train[:, 2+len(cols_static):-4].astype(int)
	y_train = train[:, -1].astype(int)

	id_test = test[:,0]
	week_test = test[:,1]
	x_static_test = test[:, 2:2+len(cols_static)].astype(float)
	x_sequence_test = test[:, 2+len(cols_static):-4].astype(int)
	y_test_calls = test[:, -4:-1].astype(int)
	y_test = test[:, -1].astype(int)

	return x_static_train, x_sequence_train, y_train, id_train, week_train, x_static_test, x_sequence_test, y_test, id_test, week_test, y_test_calls

def reshape_seq_proper(x, n, obs):
	# assume a sequence like [x,x,x,miss,miss,miss]
	# needs to change to be like [x,miss,x,miss,x...,miss]
	# then reshape to (-1, n, obs)

	new_x = []
	for samp in x:

		samp_calls = samp[:n]
		samp_cumulative_misses = samp[n:]

		
		new_samp = []
		for c, m in zip(samp_calls, samp_cumulative_misses):
			new_samp.append(c)
			new_samp.append(m)
		new_x.append(new_samp)

	new_x = np.array(new_x)
	new_x = new_x.reshape(-1, n, obs)

	return new_x

#data format
cols_id = ['SK_ID_CURR', 'QUARTER']

cols_static = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'EXT_SOURCE_1',
                  'DAYS_BIRTH', 'NAME_EDUCATION_TYPE_Higher education', 'CODE_GENDER',
                  'NAME_INCOME_TYPE_Pensioner', 'DAYS_EMPLOYED', 'ORGANIZATION_TYPE_XNA',
                  'FLOORSMAX_AVG', 'FLOORSMAX_MEDI', 'FLOORSMAX_MODE',
                  'EMERGENCYSTATE_MODE_No', 'HOUSETYPE_MODE_block of flats', 'AMT_GOODS_PRICE',
                  'REGION_POPULATION_RELATIVE', 'ELEVATORS_AVG', 'ELEVATORS_MEDI',
                  'FLOORSMIN_AVG', 'FLOORSMIN_MEDI', 'WALLSMATERIAL_MODE_Panel',
                  'LIVINGAREA_AVG', 'LIVINGAREA_MEDI', 'FLOORSMIN_MODE', 'TOTALAREA_MODE',
                  'ELEVATORS_MODE', 'NAME_CONTRACT_TYPE_Cash loans', 'LIVINGAREA_MODE', 'AMT_CREDIT']

cols_sequence_x = ['x'+str(i) for i in range(1,4)] + ['mis'+str(i) for i in range(1,4)]

cols_sequence_y = ['y1', 'y2', 'y3']

#switch depending on model
# cols_label = ['TARGET']
cols_label = ['label']

prior_attention = 'M'

if prior_attention != 'M' and prior_attention != 'H':
	raise ValueError

#lstm model values
x_len = 3
y_len = 3

N_TRAIN_DAYS = [3]

precision_list = []
recall_list = []
auc_pr_list = []

tpr_list = []
fpr_list = []
auc_roc_list = []
threshold_roc_list = []

epochs = [20]
OBSERVATIONS = 2
BATCH_SIZE = 128
THRESHOLD = 0.902
VAL_SPLIT_SIZE = 3

saved_preds_and_test = {'predictions':[],'test':[]}

for n_index, n in enumerate(N_TRAIN_DAYS):

	#data
	data_df = pd.read_csv('data/processed/train_and_test_x%iy%i_H.csv'%(x_len, y_len))
	data_df = data_df.drop(columns=['TARGET'])

	#extract values from data
	x_static_train, x_sequence_train, y_train, id_train, week_train, x_static_test, x_sequence_test, y_test, id_test, week_test, y_test_calls = restructure_data(data_df)

	split_size = y_train.shape[0] // VAL_SPLIT_SIZE
	print("split size,", split_size, "\n")

	x_static_train_subset, x_sequence_train_subset, y_train_subset = x_static_train[:-split_size], x_sequence_train[:-split_size], y_train[:-split_size]
	x_static_val_subset, x_sequence_val_subset, y_val_subset = x_static_train[-split_size:], x_sequence_train[-split_size:], y_train[-split_size:]

	# Scale features as normal distributions
	feature_scaler = myScaler().fit(x_static_train_subset)
	x_static_train_subset = feature_scaler.transform(x_static_train_subset)
	x_static_val_subset = feature_scaler.transform(x_static_val_subset)
	x_static_test = feature_scaler.transform(x_static_test)

	x_train_merged = np.concatenate([x_static_train_subset, x_sequence_train_subset],axis=1)

	#sampling information to resample data set https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html
	sm = SMOTE(random_state=42)

	print ("Train stats")
	print( y_train)
	print (y_train.shape)
	print (y_train.sum())

	print ("Test stats")
	print (y_test)
	print (y_test.shape)
	print (y_test.sum())

	print (x_train_merged.dtype)

	print (x_train_merged.dtype)

	x_train_merged, y_train_subset = sm.fit_sample(x_train_merged, y_train_subset)
	x_static_train_subset, x_sequence_train_subset = x_train_merged[:, :len(cols_static)], x_train_merged[:, len(cols_static):]

	x_sequence_test_original = x_sequence_test

	x_sequence_train_subset = reshape_seq_proper(x_sequence_train_subset, n, OBSERVATIONS)

	x_sequence_val_subset = reshape_seq_proper(x_sequence_val_subset, n, OBSERVATIONS)
	x_sequence_test = reshape_seq_proper(x_sequence_test, n, OBSERVATIONS)

	print(x_sequence_train_subset[:1])
	print(x_sequence_test[:1])

	num_static_observations = x_static_train_subset.shape[1]
	num_lstm_units = 64
	num_dense_static = 100
	num_dense_penultimate = 16

	model = models.get_lstm_plus_static_model(num_lstm_units, n, 
		OBSERVATIONS, num_static_observations, num_dense_static,
		num_dense_penultimate)

	test_prauc_per_epoch = []
	very_inner_p = []
	very_inner_r = []

	test_auroc_per_epoch = []
	very_inner_tpr = []
	very_inner_fpr = []
	threshold_roc_per_epoch = []

	# We will take the max over these 
	val_prauc_per_epoch = []
	val_auroc_per_epoch = []



	EPOCHS = epochs[n_index]
	for i in range(EPOCHS):

		model.fit([x_sequence_train_subset, x_static_train_subset], y_train_subset, epochs=1, batch_size=BATCH_SIZE)

		# Validation set
		predictions = model.predict([x_sequence_val_subset, x_static_val_subset])
		p, r, threshold = precision_recall_curve(y_val_subset, predictions)
		auc_pr = auc(r, p)
		val_prauc_per_epoch.append(auc_pr)
		print('val aucpr',auc_pr)

		fpr, tpr, threshold = roc_curve(y_val_subset, predictions)
		auc_roc = auc(fpr, tpr)
		val_auroc_per_epoch.append(auc_roc)
		print('val auc_roc',auc_roc)

		# Test set
		predictions = model.predict([x_sequence_test, x_static_test])
		p, r, threshold = precision_recall_curve(y_test, predictions)
		auc_pr = auc(r, p)


		very_inner_r.append(r)
		very_inner_p.append(p)
		test_prauc_per_epoch.append(auc_pr)


		fpr, tpr, threshold = roc_curve(y_test, predictions)
		auc_roc = auc(fpr, tpr)
		
		very_inner_fpr.append(fpr)
		very_inner_tpr.append(tpr)
		test_auroc_per_epoch.append(auc_roc)
		threshold_roc_per_epoch.append(threshold)


	saved_preds_and_test['predictions']=predictions
	saved_preds_and_test['test'] = y_test
	saved_preds_and_test['y_test_calls'] = y_test_calls
	saved_preds_and_test['x_test_calls'] = x_sequence_test_original.reshape(-1,6)[:, :3]

	predictions_thresholded = threshold_predicitons(predictions, THRESHOLD)

	correct_positives = predictions_thresholded & y_test

	print("True Positives:",correct_positives.sum())

	calls_of_correct_positives = get_calls_of_correct_positives(correct_positives, y_test_calls)
	missed_calls_of_correct_positives = abs(1 - calls_of_correct_positives)
	print(missed_calls_of_correct_positives.shape)

	missed_calls = missed_calls_of_correct_positives.sum()
	print("Number of misses:",missed_calls)

	false_positives = get_false_positives(predictions_thresholded, y_test)
	print("Number of false positives:",false_positives.sum())

	true_negatives = np.logical_not(predictions_thresholded|y_test).astype(int)
	print("Number of true negatives:",true_negatives.sum())

	x_sequence_test = x_sequence_test.reshape(-1,6)
	predictions = predictions.reshape(-1,1)
	y_test = y_test.reshape(-1,1)
	for eh in [x_sequence_test, x_static_test, y_test_calls, y_test, predictions]:
		print(eh.shape)
	print_out = np.concatenate([x_sequence_test, x_static_test, y_test_calls, y_test, predictions],axis=1)
	print_out_df = pd.DataFrame(print_out, columns=cols_sequence_x+cols_static+cols_sequence_y+cols_label+['pred'])
	
	# p, r, threshold = precision_recall_curve(y_test, predictions)

	# Take max over validation set, not test set -- 'simulating' blindness to test set
	# PR
	ind = np.argmax(val_prauc_per_epoch)
	bestp = very_inner_p[ind]
	bestr = very_inner_r[ind]
	bestauc_pr = test_prauc_per_epoch[ind]


	# ROC
	ind = np.argmax(val_auroc_per_epoch)
	best_tpr = very_inner_tpr[ind]
	best_fpr = very_inner_fpr[ind]
	bestauc_roc = test_auroc_per_epoch[ind]
	best_threshold_roc = threshold_roc_per_epoch[ind]


	precision_list.append(bestp)
	recall_list.append(bestr)
	auc_pr_list.append(bestauc_pr)

	tpr_list.append(best_tpr)
	fpr_list.append(best_fpr)
	auc_roc_list.append(bestauc_roc)
	threshold_roc_list.append(best_threshold_roc)


#helper functions
def make_serializable(d):
	keys_we_care_about = ['pr', 'roc']

	for top_key in keys_we_care_about:
		for key in d[top_key]:
			for i,array in enumerate(d[top_key][key]):
				d[top_key][key][i] = array.tolist()

def make_serializable_simp(d):
	for key in d:
		d[key] = d[key].tolist()

#retrieve plots
import json
fname = 'json_data/lstm_final_'+prior_attention+'_preds.json'
make_serializable_simp(saved_preds_and_test)
f = open(fname, 'w')
json.dump(saved_preds_and_test,f)
f.close()


plot_data = {'pr':{}, 'roc':{}}

plot_data['pr']['r']=recall_list
plot_data['pr']['p']=precision_list
plot_data['pr']['auc']=auc_pr_list

plot_data['roc']['fpr']=fpr_list
plot_data['roc']['tpr']=tpr_list
plot_data['roc']['auc']=auc_roc_list
plot_data['roc']['threshold']=threshold_roc_list

plot_data['n']=N_TRAIN_DAYS

make_serializable(plot_data)

fname = 'json_data/lstm_final_'+prior_attention+'_plotdata.json'
f = open(fname, 'w')
json.dump(plot_data, f)
f.close()

for i in range(len(N_TRAIN_DAYS)):
	plt.plot(recall_list[i], precision_list[i], label='k='+str(N_TRAIN_DAYS[i])+', auc:'+str(auc_pr_list[i]))
plt.legend()
plt.title('LSTM+Static Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('plots/prcurve_lstm_'+prior_attention+'.png')
plt.show()

for i in range(len(N_TRAIN_DAYS)):
	plt.plot(fpr_list[i], tpr_list[i], label='k='+str(N_TRAIN_DAYS[i])+', auc:'+str(auc_roc_list[i]))
	print("LENGTH OF THRESHOLD LIST:",len(threshold_roc_list[i]))
	for j in range(len(threshold_roc_list[i])):
		x = len(threshold_roc_list[i])/15
		if j%x==0:
			plt.annotate(str(threshold_roc_list[i][j]),(fpr_list[i][j], tpr_list[i][j]))

plt.legend()
plt.title('LSTM+Static ROC Curve')
lw = 2
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.savefig('plots/roccurve_lstm_'+prior_attention+'.png')
plt.show()



