import os
import pandas as pd
import numpy as np
from os import listdir
from sklearn.model_selection import train_test_split

classes = ['PROD_NAME', 'PROD_NUM', 'UNIT_PRICE', 'PROD_BARCODE_NUMBER', 'OTHER']

df = pd.DataFrame(columns=['PRODUCTS', '','','',''])
classes_series = pd.Series(classes, index=df.columns)
df = df.append(classes_series, ignore_index=True)
df.to_csv('test1.csv', index=False)

PATH = r'/Data/Raw Data'
list_files = listdir(PATH)

data_list = []
i = 0
for j in range(len(list_files)):
    print(i + len(list_files), 'of', (len(list_files)), "files remaining.")
    dataset_filename = os.listdir(PATH)[j]
    print(dataset_filename)
    dataset_path = os.path.join("../../..", PATH, dataset_filename)
    data = pd.read_csv(dataset_path, error_bad_lines=False, engine='c', encoding="UTF-8", low_memory=False,
                       skiprows=1)
    data = data[['PROD_NAME', 'PROD_NUM', 'UNIT_PRICE']]
    data.drop_duplicates(subset=['PROD_NAME'], inplace=True)
    data.drop_duplicates(subset=['PROD_NUM'], inplace=True)
    data.drop_duplicates(subset=['UNIT_PRICE'], inplace=True)
    data_list.append(data)
    i -= 1

dfc = pd.concat(data_list)
print(df.shape)
dfc.to_csv('test.csv', index=False)
