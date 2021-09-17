import os
import pandas as pd

PATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\test'
classes = ['BOOL', 'DATE_TIME', 'DATE', 'URL', 'PROD_PHOTO_URL', 'PDF', 'LANGUAGE_ID', 'COUNT', 'INTERNAL_ID',
           'PROD_CAT_ID', 'PROD_TYPE_ID', 'DESC_LONG', 'MANUFAC_ID', 'PROD_NAME', 'PROD_NUM', 'TITLE',
           'META_DESCRIPTION']

def evaluate_classes(path):
    df = pd.read_csv(path, error_bad_lines=False, engine='c', encoding='ISO-8859-1',
                     low_memory=False)
    i = 0
    for n in range(len(classes)):
        compared = df['new_target'] == classes[n]
        #print(classes[n])
        for m in range(len(compared)):
            if compared[m]:
                if df['compared'][m]:
                    print(df['new_target'][m])
                    i += 1
    print(i, len(compared))



def evaluate_classes_2(path):
    df = pd.read_csv(path, error_bad_lines=False, engine='c', encoding='ISO-8859-1',
                     low_memory=False)


    BOOL = df[df['new_target'] == 'BOOL']
    BOOL_ACC = (BOOL.compared.sum()/len(BOOL))*100
    if len(BOOL != 0):
        print('BOOL', BOOL_ACC)
    else:
        print('BOOL Empty')

    DATE_TIME = df[df['new_target'] == 'DATE_TIME']
    DATE_TIME_ACC = (DATE_TIME.compared.sum() / len(DATE_TIME) * 100)
    if len(DATE_TIME != 0):
        print('DATE_TIME', DATE_TIME_ACC)
    else:
        print('DATE_TIME Empty')

    DATE = df[df['new_target'] == 'DATE']
    DATE_ACC = (DATE.compared.sum() / len(DATE) * 100)
    if len(DATE != 0):
        print('DATE', DATE_ACC)
    else:
        print('DATE Empty')

    URL = df[df['new_target'] == 'URL']
    URL_ACC = (URL.compared.sum() / len(URL) * 100)
    if len(URL != 0):
        print('URL', URL_ACC)
    else:
        print('URL Empty')

    PROD_PHOTO_URL = df[df['new_target'] == 'PROD_PHOTO_URL']
    PROD_PHOTO_URL_ACC = (PROD_PHOTO_URL.compared.sum() / len(PROD_PHOTO_URL) * 100)
    if len(PROD_PHOTO_URL != 0):
        print('PROD_PHOTO_URL', PROD_PHOTO_URL_ACC)
    else:
        print('PROD_PHOTO_URL Empty')

    PDF = df[df['new_target'] == 'PDF']
    PDF_ACC = (PDF.compared.sum() / len(PDF) * 100)
    if len(PDF != 0):
        print('PDF', PDF_ACC)
    else:
        print('PDF Empty')

    LANGUAGE_ID = df[df['new_target'] == 'LANGUAGE_ID']
    LANGUAGE_ID_ACC = (LANGUAGE_ID.compared.sum() / len(LANGUAGE_ID) * 100)
    if len(LANGUAGE_ID != 0):
        print('LANGUAGE_ID', LANGUAGE_ID_ACC)
    else:
        print('LANGUAGE_ID Empty')

    COUNT = df[df['new_target'] == 'COUNT']
    COUNT_ACC = (COUNT.compared.sum() / len(COUNT) * 100)
    if len(COUNT != 0):
        print('COUNT', COUNT_ACC)
    else:
        print('COUNT Empty')

    INTERNAL_ID = df[df['new_target'] == 'INTERNAL_ID']
    INTERNAL_ID_ACC = (INTERNAL_ID.compared.sum() / len(INTERNAL_ID) * 100)
    if len(INTERNAL_ID != 0):
        print('INTERNAL_ID', INTERNAL_ID_ACC)
    else:
        print('INTERNAL_ID Empty')

    PROD_CAT_ID = df[df['new_target'] == 'PROD_CAT_ID']
    PROD_CAT_ID_ACC = (PROD_CAT_ID.compared.sum() / len(PROD_CAT_ID) * 100)
    if len(PROD_CAT_ID != 0):
        print('PROD_CAT_ID', PROD_CAT_ID_ACC)
    else:
        print('PROD_CAT_ID Empty')

    PROD_TYPE_ID = df[df['new_target'] == 'PROD_TYPE_ID']
    PROD_TYPE_ID_ACC = (PROD_TYPE_ID.compared.sum() / len(PROD_TYPE_ID) * 100)
    if len(PROD_TYPE_ID != 0):
        print(PROD_TYPE_ID_ACC)
    else:
        print('PROD_TYPE_ID Empty')

    DESC_LONG = df[df['new_target'] == 'DESC_LONG']
    DESC_LONG_ACC = (DESC_LONG.compared.sum() / len(DESC_LONG) * 100)
    if len(DESC_LONG != 0):
        print('PROD_TYPE_ID', DESC_LONG_ACC)
    else:
        print('DESC_LONG Empty')

    MANUFAC_ID = df[df['new_target'] == 'MANUFAC_ID']
    MANUFAC_ID_ACC = (MANUFAC_ID.compared.sum() / len(MANUFAC_ID) * 100)
    if len(MANUFAC_ID != 0):
        print('MANUFAC_ID', MANUFAC_ID_ACC)
    else:
        print('MANUFAC_ID Empty')

    PROD_NAME = df[df['new_target'] == 'PROD_NAME']
    PROD_NAME_ACC = (PROD_NAME.compared.sum() / len(PROD_NAME) * 100)
    if len(PROD_NAME != 0):
        print('PROD_NAME', PROD_NAME_ACC)
    else:
        print('PROD_NAME Empty')

    PROD_NUM = df[df['new_target'] == 'PROD_NUM']
    PROD_NUM_ACC = (PROD_NUM.compared.sum() / len(PROD_NUM)* 100)
    if len(PROD_NUM != 0):
        print('PROD_NUM', PROD_NUM_ACC)
    else:
        print('PROD_NUM Empty')

    TITLE = df[df['new_target'] == 'TITLE']
    TITLE_ACC = (TITLE.compared.sum() / len(TITLE) * 100)
    if len(TITLE != 0):
        print('TITLE', TITLE_ACC)
    else:
        print('TITLE Empty')


    META_DESCRIPTION = df[df['new_target'] == 'META_DESCRIPTION']
    META_DESCRIPTION_ACC = (META_DESCRIPTION.compared.sum() / len(META_DESCRIPTION) * 100)
    if len(META_DESCRIPTION != 0):
        print('META_DESCRIPTION', META_DESCRIPTION_ACC)
    else:
        print('META_DESCRIPTION Empty')






list_files = os.listdir(PATH)
i = 0
for j in range(len(list_files)-(len(list_files)-10)):
    print(i + len(list_files), 'of', (len(list_files)))
    dataset_filename = os.listdir(PATH)[j]
    dataset_path = os.path.join("../..", PATH, dataset_filename)
    print(dataset_filename)
    evaluate_classes_2(dataset_path)
    i -= 1


# my_list = ['foo', 'fob', 'faz', 'funk']
# string = 'bar'
# list2 = list(map(lambda orig_string: orig_string + string, my_list))