### Early data engineering script - transforms data into 2d column for training, also splits train/test sets ###

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

PATH = r'C:\Users\mail\Downloads\data\Parcellet\models_revamp\data\NAME_COLOR.csv'

df = pd.read_csv(PATH, error_bad_lines=False, engine='c', encoding="ISO-8859-15", low_memory=False)

print(df.shape)

df_train, df_test = train_test_split(df, test_size=0.2)

print(df_train.shape)
print(df_test.shape)


def cols2DF(data, name):
    # name = dataset_filename
    columns = data.columns
    dfList = []
    for i in columns:
        x = data[i]
        x = pd.DataFrame(x)
        x['Header'] = i
        x.rename(columns={i: 'Data'}, inplace=True)
        dfList.append(x)
        y = pd.concat(dfList)
        y.dropna(axis=0, how='any', inplace=True)
    y.to_csv(name, index=False)

os.chdir(r'C:\Users\mail\Downloads\data\Parcellet\models_revamp\data\2D')

cols2DF(df_test, 'NAME_COLOR_test_2D.csv')

cols2DF(df_train, 'NAME_COLOR_train_2D.csv')

ros = RandomOverSampler(sampling_strategy='auto')

df_train_2D = pd.read_csv(r'C:\Users\mail\Downloads\data\Parcellet\models_revamp\data\2D'
                          r'\NAME_COLOR_train_2D.csv',
                          error_bad_lines=False, engine='c', encoding="UTF-8", low_memory=False)

x_Data = df_train_2D['Data']
y_Target = df_train_2D['Header']


Data_Oversampled, y_Target_Oversampled = ros.fit_resample(df_train_2D, y_Target)

Data_Oversampled.to_csv('NAME_COLOR_train_2D_over.csv', index=False)

print(Data_Oversampled['Header'].value_counts())

