# Concatenates all .csv files from dir into one .csv
import os

import pandas as pd

from os import listdir

DATAPATH = r'C:\Users\surface\Desktop\YouWe\MLDM\Data\2D Data'
SAVEPATH = r'C:\Users\surface\Desktop\YouWe\MLDM\Data\Concatenated Data'

files = listdir(DATAPATH)
df_list = []
i = 0

for j in range(len(files)):
    print(i)
    dataset_filename = os.listdir(DATAPATH)[j]
    dataset_path = os.path.join("..", DATAPATH, dataset_filename)
    data = pd.read_csv(DATAPATH, error_bad_lines=False, engine='c', encoding='ISO-8859-1', low_memory=False, skiprows=1)
    i += 1
    df_list.append(data)

os.chdir(SAVEPATH)
df = pd.concat(df_list)
df.dropna(axis=1, how='all', inplace=True)
df.to_csv('concat_test.csv', index=False)
