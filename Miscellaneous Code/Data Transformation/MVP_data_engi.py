### Old Data engineering script for for early DanDomain classifier MVP ###

import pandas as pd
import numpy as np

Name = r'C:\Users\mail\Desktop\Data\CSVpad 1.2 64bit\Name.csv'
Description = r'C:\Users\mail\Desktop\Data\CSVpad 1.2 64bit\Description.csv'
Category = r'C:\Users\mail\Desktop\Data\CSVpad 1.2 64bit\Category.csv'
Price = r'C:\Users\mail\Desktop\Data\CSVpad 1.2 64bit\Price.csv'
Amount = r'C:\Users\mail\Desktop\Data\CSVpad 1.2 64bit\Amount.csv'
EAN = r'C:\Users\mail\Desktop\Data\CSVpad 1.2 64bit\EAN.csv'

df_Name = pd.read_csv(Name, error_bad_lines=False, engine='c', encoding='UTF-8', low_memory=False)
df_Description = pd.read_csv(Description, error_bad_lines=False, engine='c', encoding='UTF-8', low_memory=False)
df_Category = pd.read_csv(Category, error_bad_lines=False, engine='c', encoding='UTF-8', low_memory=False)
df_Price = pd.read_csv(Price, error_bad_lines=False, engine='c', encoding='UTF-8', low_memory=False)
df_Amount = pd.read_csv(Amount, error_bad_lines=False, engine='c', encoding='UTF-8', low_memory=False)
df_EAN = pd.read_csv(EAN, error_bad_lines=False, engine='c', encoding='UTF-8', low_memory=False)

df_Name = df_Name.sample(n=57889)
df_Description = df_Description.sample(n=57889)
df_Category = df_Category.sample(n=57889)
df_Price = df_Price.sample(n=57889)
df_Amount = df_Amount.sample(n=57889)
df_EAN = df_EAN.sample(n=57889)

cols = df_Name.columns.to_list() + df_Description.columns.to_list() + df_Category.columns.to_list() + df_Price.columns.to_list() + df_Amount.columns.to_list() + df_EAN.columns.to_list()
data = [df_Name, df_Description, df_Category, df_Price, df_Amount, df_EAN]
df = np.concatenate(data, axis=1)
df = pd.DataFrame(df, columns=cols)

df.to_csv('6_class_MVP_dataset.csv', index=False)