import os
import pandas as pd

PATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\test'
classes = ['BOOL', 'DATE_TIME', 'DATE', 'URL', 'PROD_PHOTO_URL', 'PDF', 'LANGUAGE_ID', 'COUNT', 'INTERNAL_ID',
           'PROD_CAT_ID', 'PROD_TYPE_ID', 'DESC_LONG', 'MANUFAC_ID', 'PROD_NAME', 'PROD_NUM', 'TITLE',
           'META_DESCRIPTION']

def evaluate_classes(path):
    df = pd.read_csv(path, error_bad_lines=False, engine='c', encoding='ISO-8859-1',
                     low_memory=False)
    print(df['new_target'])


list_files = os.listdir(PATH)
i = 0
for j in range(len(list_files)):
    print(i + len(list_files), 'of', (len(list_files)))
    dataset_filename = os.listdir(PATH)[j]
    dataset_path = os.path.join("../..", PATH, dataset_filename)
    print(dataset_filename)
    evaluate_classes(dataset_path)
    i -= 1