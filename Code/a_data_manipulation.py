import pandas as pd
import numpy as np
import time
from collections import defaultdict

#helper function to reduce memory usage https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

#helper function to be used with reduce_mem_usage
def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df

#determine the number of people who defaulted in their loan
def determine_default(sequence):
    default = []

    for key in sequence:
        string = sequence[key][1]

        for char in string:
            if char == '5':
                default.append(key)

    final = {}
    for key in default:
        final.update({key: sequence[key]})

    return final

#helper fn to be used with get_atention_sequence
def get_attention_for_previous_months(payment_profile):
    #need to change
    for char in payment_profile:
        # create H flag
        if char == '5':
            return 'HHH'

    return 'MMM'

# create attention sequence for previous 6 months
def get_attention_sequence(payment_profile):
    MONTHS = 3
    payment_profile = payment_profile[1]
    overtime = MONTHS - len(payment_profile) % MONTHS

    if overtime != 0:
        for i in range(overtime):
            payment_profile = '0' + payment_profile

    atten = ''
    
    for i in range(int(len(payment_profile) / MONTHS)):
        index = i * 3

        previous_months = payment_profile[index:index + MONTHS]
        attention = get_attention_for_previous_months(previous_months)
        atten = atten + attention
    
    for i in range(overtime):
        atten = atten[1:]

    return atten

#import balance data
print('-' * 80)
print('bureau')
df = import_data('data/raw/bureau_balance.csv')

print(df.shape)

#time process
tic = time.perf_counter()

#format data into usable form 
# 1. change all instances of X with 0
# 2. delete all rows with a C
# 3. group all similar ID numbers together into one string
# 4. eliminate all loans with perfect pay records

df = df.replace('X', '0')
df = df[df.STATUS != 'C']
df = df.groupby('SK_ID_BUREAU')['STATUS'].apply(''.join).reset_index()
print(df)

#iterate over dataframe and convert to dictionary, and count the number of misses
num_vals = df.shape[0]
sequence_dict = {}

count = 0
arr = df.to_numpy()

for i in range(num_vals):
    if int(arr[i][1]) != 0 and len(arr[i][1]) > 6:

        seq = arr[i][1]

        non_zeros = 0
        for char in seq:
            if char != '0':
                non_zeros += 1

        sequence_dict.update({arr[i][0]: [non_zeros, seq]})
            
        count += non_zeros

#print useful data
print("Total number of people: ", df.shape[0])
print("Total number of misses: ", count)
print("Number of people show missed at least 1 payment: ", len(sequence_dict))
print("Number of people who defaulted: ", len(determine_default(sequence_dict)))

#convert dictionary back into dataframe
df_dict = defaultdict(list)
df_dict = {'SK_ID_BUREAU': [], 'MONTHS_MISSED': [], 'TimeInProgram': [], 'AdherenceSequence': [], 'AttentionString': []}

for person in sequence_dict:
    df_dict['SK_ID_BUREAU'].append(person)
    df_dict['MONTHS_MISSED'].append(sequence_dict[person][0])
    df_dict['TimeInProgram'].append(len(sequence_dict[person][1]))
    df_dict['AdherenceSequence'].append(sequence_dict[person][1])

    atten = get_attention_sequence(sequence_dict[person])

    df_dict['AttentionString'].append(atten)

    #check to make sure everything is of the same length
    if len(atten) != len(sequence_dict[person][1]):
        print('something wrong')

#export data
df = pd.DataFrame.from_dict(df_dict)
print(df)
df.to_csv('data/intermediate_data/adherence_sequences.csv', index=False)
toc = time.perf_counter()

#time to calculate
print(f"Convert all data in {toc - tic:0.4f} seconds")