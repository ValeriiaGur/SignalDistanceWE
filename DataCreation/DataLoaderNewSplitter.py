import numpy as np
import pandas as pd
import os
import json
from copy import deepcopy
from numba import njit
from tqdm import tqdm
from tqdm._tqdm_notebook import tqdm_notebook



tqdm_notebook.pandas()

@njit(parallel=True)
def split_to_parts_with_r(a, max_length, min_length, r=None):
    new_array = split_to_parts(a, max_length, min_length)
    rpeaks_splitted = []
    index_r = 1
    rpeaks = []
    for i in r:
        if i >= index_r * max_length:
            index_r += 1
            #print("here")
            rpeaks_splitted.append(rpeaks.copy())
            rpeaks.clear()
        #print(i)
        new_i = i % max_length
        #print(new_i)
        rpeaks.append(new_i)
        
    rpeaks_splitted.append(rpeaks)
    new_array = new_array[:len(rpeaks_splitted)]
    #print(len(new_array))
    #print(len(rpeaks_splitted))
    return new_array, rpeaks_splitted


@njit(parallel=True)
def split_to_parts(a, max_length, min_length):
    num_parts = int(len(a) / max_length)
    indeces = [i * max_length for i in range(1, num_parts+1)]
    #print(indeces)
    if num_parts > 0:
        new_array = []
        prev = 0
        for i in indeces:
            new_array.append(a[prev:i])
            prev = i
        new_array.append(a[i:])
        if len(new_array[-1])<min_length:
            new_array = new_array[:-1]
    else:
        new_array = [a]
    return new_array
    

def split_long_signals(df, min_length, max_length, signal_col, rpeaks=False):
    df = deepcopy(df)
    columns = list(df.columns.difference([signal_col]))
    df = df[columns+[signal_col]]
    rows = []
    if not rpeaks:
        columns = list(df.columns.difference([signal_col]))
        df = df[columns+[signal_col]]
        df[signal_col] = df[signal_col].apply(lambda x: split_to_parts(x, max_length, min_length))
        _ = df.apply(lambda row: [rows.append([row[c] for c in columns] + [signal]) 
                         for signal in row[signal_col]], axis=1)
    else:
        columns = list(df.columns.difference([signal_col, "rpeaks"]))
        df = df[columns+[signal_col, "rpeaks"]]
        print("here1")
        df[signal_col+"rpeaks"] = df[[signal_col, "rpeaks"]].progress_apply(lambda x: split_to_parts_with_r(x[0], max_length, 
                                                                                                    min_length, x[1]), axis=1)
        print("here")
        df[signal_col] = df[signal_col+"rpeaks"].apply(lambda x: x[0])
        df["rpeaks"] = df[signal_col+"rpeaks"].apply(lambda x: x[1])
        df = df.drop([signal_col+"rpeaks"], axis=1)
        #print(df.head)
        _ = df.apply(lambda row: [rows.append([row[c] for c in columns] + [signal, rpeaks]) 
                         for signal, rpeaks in zip(row[signal_col], row["rpeaks"])], axis=1)
    #print(rows[0])
    #print(df.columns)
    
    df_new = pd.DataFrame(rows, columns=df.columns)
    return df_new


def save_df(df, array_columns, folder_name):
    df = df.reset_index(drop=True)
    if isinstance(array_columns, str):
        array_columns = [array_columns]
    rest_csv_name = os.path.join(folder_name, 'rest.csv')
    rest_df = df.drop(array_columns, axis = 1)
    os.makedirs(folder_name, exist_ok=True)
    rest_df.to_csv(rest_csv_name, index = False)
    info = {'array_cols':[]}
        
    for col in array_columns:
        total = np.concatenate(df[col])
        offsets = df[col].apply(lambda x : x.shape[0]).cumsum()
        total_name = os.path.join(folder_name, f'{col}_total.npy')
        offsets_name = os.path.join(folder_name, f'{col}_offsets.npy')
        np.save(total_name, total)
        np.save(offsets_name, offsets)
        info['array_cols'].append(col)
    info_name = os.path.join(folder_name, 'info.json')
    with open(info_name, 'w') as fout:
        json.dump(info, fout)
        
        
def load_df(folder_name):
    csv_name = os.path.join(folder_name, 'rest.csv')
    df = pd.read_csv(csv_name)
    info_name = os.path.join(folder_name, 'info.json')
    with open(info_name, 'r') as fin:
        info = json.load(fin)
    for col in info['array_cols']:
        total_name = os.path.join(folder_name, f'{col}_total.npy')
        offsets_name = os.path.join(folder_name, f'{col}_offsets.npy')
        total = np.load(total_name)
        offsets = np.load(offsets_name)
        df[col] = pd.Series([total[offsets[i-1] if i > 0 else 0:offsets[i]] for i in range(offsets.shape[0])])
    return df