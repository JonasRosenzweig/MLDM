### Testing scrambling of dataframe using df.sample(frac=x) ###
import os
import pandas as pd

PATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\test\abrideustyr.csv'

df = pd.read_csv(PATH, error_bad_lines=False, engine='c', encoding='ISO-8859-1',
                     low_memory=False, skiprows=1)

df = df.sample(frac=0.05, random_state=1)
df = df.sample(frac=1, random_state=1)
os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\accuracies')
df.to_csv('test.csv', index=False)