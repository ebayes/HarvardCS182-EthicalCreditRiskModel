from keras.layers import Input, LSTM, Dense, concatenate
from keras.models import Model, Sequential
import keras.backend as K

def get_lstm_plus_static_model(num_lstm_units, num_call_days, 
	num_lstm_observations_per_day, num_static_observations, num_dense_static,
	num_dense_penultimate):

	# Sequential LSTM layer
	lstm_input = Input(shape=(num_call_days, num_lstm_observations_per_day))
	lstm_layer = LSTM(num_lstm_units)
	lstm_outputs = lstm_layer(lstm_input)

	# Totally separate Static feature layer
	static_input = Input(shape=(num_static_observations,))
	static_layer = Dense(num_dense_static)
	static_output = static_layer(static_input)

	# Concatenate LSTM output and static features layer output
	merged_vector = concatenate([lstm_outputs, static_output], axis=-1)

	# Create one more dense layer with the outputs from combined lstm+static layers
	dense_1 = Dense(num_dense_penultimate, activation='sigmoid')(merged_vector)

	# Single neuron layer for the predictions
	predictions = Dense(1, activation='sigmoid')(dense_1)

	# Define a trainable model linking the merged inputs to the predictions
	model = Model(inputs=[lstm_input, static_input], outputs=predictions)
	model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_crossentropy'])
	model.summary()

	return model

def weighted_bce(yTrue,yPred):
	print(4*yTrue*K.log(yPred) + (1 - yTrue)* K.log(1 - yPred))
	return 4*yTrue*K.log(yPred) + (1 - yTrue)* K.log(1 - yPred)


def get_lstm_plus_static_model_w(num_lstm_units, num_call_days, 
	num_lstm_observations_per_day, num_static_observations, num_dense_static,
	num_dense_penultimate):

	# Sequential LSTM layer
	lstm_input = Input(shape=(num_call_days, num_lstm_observations_per_day))
	lstm_layer = LSTM(num_lstm_units)
	lstm_outputs = lstm_layer(lstm_input)

	# Totally separate Static feature layer
	static_input = Input(shape=(num_static_observations,))
	static_layer = Dense(num_dense_static)
	static_output = static_layer(static_input)

	# Concatenate LSTM output and static features layer output
	merged_vector = concatenate([lstm_outputs, static_output], axis=-1)

	# Create one more dense layer with the outputs from combined lstm+static layers
	dense_1 = Dense(num_dense_penultimate, activation='sigmoid')(merged_vector)

	# Single neuron layer for the predictions
	predictions = Dense(1, activation='sigmoid')(dense_1)

	# Define a trainable model linking the merged inputs to the predictions
	model = Model(inputs=[lstm_input, static_input], outputs=predictions)
	model.compile(optimizer='adam',
              loss=weighted_bce,
              metrics=['binary_crossentropy'])
	model.summary()

	return model


def get_lstm_plus_static_model_no_penultimate(num_lstm_units, num_call_days, 
	num_lstm_observations_per_day, num_static_observations, num_dense_static):

	# Sequential LSTM layer
	lstm_input = Input(shape=(num_call_days, num_lstm_observations_per_day))
	lstm_layer = LSTM(num_lstm_units)
	lstm_outputs = lstm_layer(lstm_input)

	# Totally separate Static feature layer
	static_input = Input(shape=(num_static_observations,))
	static_layer = Dense(num_dense_static)
	static_output = static_layer(static_input)

	# Concatenate LSTM output and static features layer output
	merged_vector = concatenate([lstm_outputs, static_output], axis=-1)

	# Create one more dense layer with the outputs from combined lstm+static layers
	predictions = Dense(1, activation='sigmoid')(merged_vector)


	# Define a trainable model linking the merged inputs to the predictions
	model = Model(inputs=[lstm_input, static_input], outputs=predictions)
	model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_crossentropy'])
	model.summary()

	return model

# def get_lstm_plus_static_model_no_dense_static(num_lstm_units, num_call_days, 
# 	num_lstm_observations_per_day, num_static_observations, num_dense_penultimate):

# 	# Sequential LSTM layer
# 	lstm_input = Input(shape=(num_call_days, num_lstm_observations_per_day))
# 	lstm_layer = LSTM(num_lstm_units)
# 	lstm_outputs = lstm_layer(lstm_input)

# 	# Totally separate Static feature layer
# 	static_input = Input(shape=(num_static_observations,))
# 	# static_layer = Dense(num_dense_static)
# 	# static_output = static_layer(static_input)

# 	# Concatenate LSTM output and static features layer output
# 	merged_vector = concatenate([lstm_outputs, static_input], axis=-1)

# 	# Create one more dense layer with the outputs from combined lstm+static layers
# 	dense_1 = Dense(num_dense_penultimate, activation='sigmoid')(merged_vector)

# 	# Single neuron layer for the predictions
# 	predictions = Dense(1, activation='sigmoid')(dense_1)


	# Define a trainable model linking the merged inputs to the predictions
	# model = Model(inputs=[lstm_input, static_input], outputs=predictions)
	# model.compile(optimizer='adam',
 #              loss='binary_crossentropy',
 #              metrics=['binary_crossentropy'])
	# model.summary()

	# return model	


def get_lstm_single_layer_model(num_lstm, n, observations):
	model = Sequential()
	model.add(LSTM(num_lstm, input_shape=(n, observations)))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy'])
	model.summary()
	return model


def get_lstm_plus_dense_model(lstm_size, num_dense, num_days, observations):
	model = Sequential()
	model.add(LSTM(lstm_size, input_shape=(num_days, observations)))
	# model.add(LSTM(lstmsize, input_shape=(n, OBSERVATIONS), return_sequences=True))
	# model.add(LSTM(10))
	model.add(Dense(num_dense, activation='sigmoid'))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', 
		metrics=['binary_crossentropy'])
	model.summary()
	return model

