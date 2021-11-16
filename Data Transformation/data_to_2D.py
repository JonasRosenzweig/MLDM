# transforms data into 2 column .csv: 'Data' and 'Header" for training
# gets rid of rows with NAN fields

import os
import pandas as pd
from os import listdir

PATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\MVP\transform'
SAVEPATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\MVP'

list_files = listdir(PATH)
print("Starting transform of all files in", PATH)
i = 0
for j in range(len(list_files)):
    print(i + len(list_files),  'of', (len(list_files)), "files remaining.")
    dataset_filename = os.listdir(PATH)[j]
    dataset_path = os.path.join("../..", PATH, dataset_filename)
    data = pd.read_csv(dataset_path, error_bad_lines=False, engine='c', encoding = "ISO-8859-1", low_memory=False)
    # sample used to randomly sample a percentage of rows to reduce memory load
    #data = data.sample(frac=0.1, random_state=1)
    i -=1

    def removePunct(name):
        name = name.replace(" ","_")
        name = name.replace(".","_")
        return name

    def cols2DF(data):
        #name = dataset_filename
        columns = data.columns
        dfList=[]
        for i in columns:
            x = data[i]
            x = pd.DataFrame(removePunct(x))
            x['Header']= i
            x.rename(columns={i:'Data'}, inplace=True)
            dfList.append(x)
            y = pd.concat(dfList)
            y.dropna(axis=0, how='any', inplace=True)
        y.to_csv('6_class_MVP_dataset_v3_unknowns_no_duplicates_2D.csv', index=False)

    os.chdir(SAVEPATH)
    cols2DF(data)

print("0 of", len(list_files), "files remaining. \nDone Transforming all files.")
