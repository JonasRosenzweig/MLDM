import os
import pandas as pd

PATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\MVP'
CSV = '6_class_MVP_dataset_v3_unknowns.csv'
TEST = os.path.join(PATH, CSV)
df = pd.read_csv(TEST, error_bad_lines=False, engine='c', encoding='ISO-8859-15', low_memory=False)

df_cols = list(df.columns)
df_list = []

for i in range(len(df_cols)):
    df_select = df[df_cols[i]]
    df_select = df_select.dropna()
    df_select = df_select.drop_duplicates()
    #df_select = df_select.sample(1087)
    df_select = df_select.reset_index(drop=True)
    df_list.append(df_select)
    print(df_cols[i], ':', len(df_select.index), 'unique rows')
df_reduced = pd.DataFrame(df_list)
df_reduced = df_reduced.T
df_reduced.head(1087)
os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\MVP')
df_reduced.to_csv('6_class_MVP_dataset_v3_unknowns_no_duplicates.csv', index=False)