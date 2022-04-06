### Full data transformation - all functions ###

# packages
import os
# os functions - save directory, recent file lookup
import pandas as pd
# for loading the data feed .csv into a pandas DataFrame
from os import listdir
# for path directory finding
from sklearn.model_selection import train_test_split
# for splitting of test/train datasets for training and validation
from imblearn.over_sampling import RandomOverSampler
# for oversampled training dataset
from imblearn.under_sampling import RandomUnderSampler
# for undersampled training dataset

# takes a string and removes punctuation and fixed danish encoding errors
def remove_punct(string):
    string = string.replace(" ","_")
    string = string.replace(".","_")
    string = string.replace("Ã¥", "å")
    string = string.replace("Ã¦", "æ")
    string = string.replace("Ã¸", "ø")
    return string

# takes a pandas df and transforms into 2D by column for Data and Headers
def df_to_2D(df):
    columns = df.columns
    dfList = []
    for i in columns:
        data = df[i]
        data = pd.DataFrame(remove_punct(data))
        data['Header'] = i
        data.rename(columns={i:'Data'}, inplace=True)
        dfList.append(data)
        df_2D = pd.concat(dfList)
        df_2D.dropna(axis=0, how='any', inplace=True)
    return df_2D

# takes a path with .csv files and an iterator j and returns a pd DataFrame
def csv_to_df(path, j):
    dataset_filename = os.listdir(path)[j]
    dataset_path = os.path.join("../..", path, dataset_filename)
    df = pd.read_csv(dataset_path, error_bad_lines=False, engine='c', encoding='UTF-8',
                     low_memory=False)
    return df, dataset_filename

# takes a directory of .csv files, a savepath, a test/train ratio and a save boolean
# transforms data into 2D  .csv: 'Data' and 'Header" columns for training
# splits train/test data; saves 3.csv (test, tran, nosplit) and returns 3 dfs and the filename
def split_train_2D(path, savepath, size, save):
    list_files = listdir(path)
    print("Begin transform for all files in", path)
    i = 0
    for j in range(len(list_files)):
        df = csv_to_df(path, j)[0]
        dataset_filename = csv_to_df(path, j)[1]
        df_train, df_test = train_test_split(df, test_size=size)
        df_train = df_to_2D(df_train)
        df_test = df_to_2D(df_test)
        os.chdir(savepath)
        df_2D = df_to_2D(df)
        name = ('full_2D_no_sample_' + dataset_filename)
        train_name = ('train_2D_no_sample_' + dataset_filename)
        test_name = ('test_2D_no_sample_' + dataset_filename)
        if save:
            df_2D.to_csv(name, index=False)
            df_train.to_csv(train_name, index=False)
            df_test.to_csv(test_name, index=False)
            print("Files saved in", savepath)
    return (df_2D, df_train, df_test, dataset_filename)

# takes a directory of .csv files, a savepath, a test/train ratio and a save boolean
# transforms data into 2 column .csv: 'Data' and 'Header" for training
# splits train/test data and oversamples; saves 3.csv (test, tran, nosplit)
def oversample(path, savepath, size, save):
    df, df_train, df_test, dataset_filename = split_train_2D(path, savepath, size, False)
    ros = RandomOverSampler(sampling_strategy='auto')
    data = df_train['Data']
    headers = df_train['Header']
    Data_oversampled, Header_oversampled = ros.fit_resample(df_train, headers)
    name = ('full_2D_no_sample_' + dataset_filename)
    train_name = ('train_2D_oversampled_' + dataset_filename)
    test_name = ('test_2D_no_sample_' + dataset_filename)
    if save:
        Data_oversampled.to_csv(train_name, index=False)
        print("Files saved in", savepath)
    return df_train

# takes a directory of .csv files, a savepath, a test/train ratio and a save boolean
# transforms data into 2 column .csv: 'Data' and 'Header" for training
# splits train/test data and oversamples; saves 3.csv (test, tran, nosplit)
def undersample(path, savepath, size, save):
    df, df_train, df_test, dataset_filename = split_train_2D(path, savepath, size, False)
    ros = RandomUnderSampler(sampling_strategy='auto')
    data = df_train['Data']
    headers = df_train['Header']
    Data_oversampled, Header_oversampled = ros.fit_resample(df_train, headers)
    name = ('full_2D_no_sample_' + dataset_filename)
    train_name = ('train_2D_oversampled_' + dataset_filename)
    test_name = ('test_2D_no_sample_' + dataset_filename)
    if save:
        Data_oversampled.to_csv(train_name, index=False)
        print("Files saved in", savepath)
    return df_train


# tests - working
PATH = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Training Data\Input'
SAVEPATH = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Training Data\Unsampled'
OVERPATH = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Training Data\Oversampled'
UNDERPATH = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Training Data\Undersampled'

#split_train_2D(PATH, SAVEPATH, 0.2, True)
#oversample(PATH, OVERPATH, 0.2, True)
#ndersample(PATH, UNDERPATH, 0.2, True)