import os
import pandas as pd
from itertools import chain

PATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\test'
classes = ['BOOL', 'DATE_TIME', 'DATE', 'URL', 'PROD_PHOTO_URL', 'PDF', 'LANGUAGE_ID', 'COUNT', 'INTERNAL_ID',
           'PROD_CAT_ID', 'PROD_TYPE_ID', 'PROD_WEIGHT', 'PRICE', 'CURRENCY_CODE', 'AUTHOR', 'DESC_LONG', 'MANUFAC_ID',
           'PROD_NAME', 'PROD_NUM', 'TITLE', 'META_DESCRIPTION']
BOOL_list = ['BOOL']
DATE_TIME_list = ['DATE_TIME']
DATE_list = ['DATE']
URL_list = ['URL']
PROD_PHOTO_URL_list = ['PROD_PHOTO_URL']
PDF_list = ['PDF']
LANGUAGE_ID_list = ['LANGUAGE_ID']
COUNT_list = ['COUNT']
INTERNAL_ID_list = ['INTERNAL_ID']
PROD_CAT_ID_list = ['PROD_CAT_ID']
PROD_TYPE_ID_list = ['PROD_TYPE_ID']
PROD_WEIGHT_list = ['PROD_WEIGHT']
PRICE_list = ['PRICE']
CURRENCY_CODE_list = ['CURRENCY_CODE']
AUTHOR_list = ['AUTHOR']
DESC_LONG_list = ['DESC_LONG']
MANUFAC_ID_list = ['MANUFAC_ID']
PROD_NAME_list = ['PROD_NAME']
PROD_NUM_list = ['PROD_NUM']
TITLE_list = ['TITLE']
META_DESCRIPTION_list = ['META_DESCRIPTION']

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
        BOOL_list.append(BOOL_ACC)
    else:
        print('BOOL Empty')
        BOOL_list.append('BOOL Empty')

    DATE_TIME = df[df['new_target'] == 'DATE_TIME']
    DATE_TIME_ACC = (DATE_TIME.compared.sum() / len(DATE_TIME) * 100)
    if len(DATE_TIME != 0):
        print('DATE_TIME', DATE_TIME_ACC)
        DATE_TIME_list.append(DATE_TIME_ACC)
    else:
        print('DATE_TIME Empty')
        DATE_TIME_list.append('DATE_TIME Empty')

    DATE = df[df['new_target'] == 'DATE']
    DATE_ACC = (DATE.compared.sum() / len(DATE) * 100)
    if len(DATE != 0):
        print('DATE', DATE_ACC)
        DATE_list.append(DATE_ACC)
    else:
        print('DATE Empty')
        DATE_list.append('DATE Empty')

    URL = df[df['new_target'] == 'URL']
    URL_ACC = (URL.compared.sum() / len(URL) * 100)
    if len(URL != 0):
        print('URL', URL_ACC)
        URL_list.append(URL_ACC)
    else:
        print('URL Empty')
        URL_list.append('URL Empty')

    PROD_PHOTO_URL = df[df['new_target'] == 'PROD_PHOTO_URL']
    PROD_PHOTO_URL_ACC = (PROD_PHOTO_URL.compared.sum() / len(PROD_PHOTO_URL) * 100)
    if len(PROD_PHOTO_URL != 0):
        print('PROD_PHOTO_URL', PROD_PHOTO_URL_ACC)
        PROD_PHOTO_URL_list.append(PROD_PHOTO_URL_ACC)
    else:
        print('PROD_PHOTO_URL Empty')
        PROD_PHOTO_URL_list.append('PROD_PHOTO_URL Empty')

    PDF = df[df['new_target'] == 'PDF']
    PDF_ACC = (PDF.compared.sum() / len(PDF) * 100)
    if len(PDF != 0):
        print('PDF', PDF_ACC)
        PDF_list.append(PDF_ACC)
    else:
        print('PDF Empty')
        PDF_list.append('PDF Empty')

    LANGUAGE_ID = df[df['new_target'] == 'LANGUAGE_ID']
    LANGUAGE_ID_ACC = (LANGUAGE_ID.compared.sum() / len(LANGUAGE_ID) * 100)
    if len(LANGUAGE_ID != 0):
        print('LANGUAGE_ID', LANGUAGE_ID_ACC)
        LANGUAGE_ID_list.append(LANGUAGE_ID_ACC)
    else:
        print('LANGUAGE_ID Empty')
        LANGUAGE_ID_list.append('LANGUAGE_ID Empty')

    COUNT = df[df['new_target'] == 'COUNT']
    COUNT_ACC = (COUNT.compared.sum() / len(COUNT) * 100)
    if len(COUNT != 0):
        print('COUNT', COUNT_ACC)
        COUNT_list.append(COUNT_ACC)
    else:
        print('COUNT Empty')
        COUNT_list.append('COUNT Empty')

    INTERNAL_ID = df[df['new_target'] == 'INTERNAL_ID']
    INTERNAL_ID_ACC = (INTERNAL_ID.compared.sum() / len(INTERNAL_ID) * 100)
    if len(INTERNAL_ID != 0):
        print('INTERNAL_ID', INTERNAL_ID_ACC)
        INTERNAL_ID_list.append(INTERNAL_ID_ACC)
    else:
        print('INTERNAL_ID Empty')
        INTERNAL_ID_list.append('INTERNAL_ID Empty')

    PROD_CAT_ID = df[df['new_target'] == 'PROD_CAT_ID']
    PROD_CAT_ID_ACC = (PROD_CAT_ID.compared.sum() / len(PROD_CAT_ID) * 100)
    if len(PROD_CAT_ID != 0):
        print('PROD_CAT_ID', PROD_CAT_ID_ACC)
        PROD_CAT_ID_list.append(PROD_CAT_ID_ACC)
    else:
        print('PROD_CAT_ID Empty')
        PROD_CAT_ID_list.append('PROD_CAT_ID Empty')

    PROD_TYPE_ID = df[df['new_target'] == 'PROD_TYPE_ID']
    PROD_TYPE_ID_ACC = (PROD_TYPE_ID.compared.sum() / len(PROD_TYPE_ID) * 100)
    if len(PROD_TYPE_ID != 0):
        print(PROD_TYPE_ID_ACC)
        PROD_TYPE_ID_list.append(PROD_TYPE_ID_ACC)
    else:
        print('PROD_TYPE_ID Empty')
        PROD_TYPE_ID_list.append('PROD_TYPE_ID Empty')

    PROD_WEIGHT = df[df['new_target'] == 'PROD_WEIGHT']
    PROD_WEIGHT_ACC = (PROD_WEIGHT.compared.sum() / len(PROD_WEIGHT) * 100)
    if len(PROD_WEIGHT != 0):
        print(PROD_WEIGHT_ACC)
        PROD_WEIGHT_list.append(PROD_WEIGHT_ACC)
    else:
        print('PROD_WEIGHT Empty')
        PROD_WEIGHT_list.append('PROD_WEIGHT Empty')

    PRICE = df[df['new_target'] == 'PROD_WEIGHT']
    PRICE_ACC = (PRICE.compared.sum() / len(PRICE) * 100)
    if len(PRICE != 0):
        print(PRICE_ACC)
        PRICE_list.append(PRICE_ACC)
    else:
        print('PRICE Empty')
        PRICE_list.append('PRICE Empty')

    CURRENCY_CODE = df[df['new_target'] == 'CURRENCY_CODE']
    CURRENCY_CODE_ACC = (CURRENCY_CODE.compared.sum() / len(CURRENCY_CODE) * 100)
    if len(CURRENCY_CODE != 0):
        print(CURRENCY_CODE_ACC)
        CURRENCY_CODE_list.append(CURRENCY_CODE_ACC)
    else:
        print('CURRENCY_CODE Empty')
        CURRENCY_CODE_list.append('CURRENCY_CODE Empty')

    AUTHOR = df[df['new_target'] == 'AUTHOR']
    AUTHOR_ACC = (AUTHOR.compared.sum() / len(AUTHOR) * 100)
    if len(AUTHOR != 0):
        print(AUTHOR_ACC)
        AUTHOR_list.append(AUTHOR_ACC)
    else:
        print('AUTHOR Empty')
        AUTHOR_list.append('AUTHOR Empty')

    DESC_LONG = df[df['new_target'] == 'DESC_LONG']
    DESC_LONG_ACC = (DESC_LONG.compared.sum() / len(DESC_LONG) * 100)
    if len(DESC_LONG != 0):
        print('DESC_LONG', DESC_LONG_ACC)
        DESC_LONG_list.append(DESC_LONG_ACC)
    else:
        print('DESC_LONG Empty')
        DESC_LONG_list.append('DESC_LONG Empty')

    MANUFAC_ID = df[df['new_target'] == 'MANUFAC_ID']
    MANUFAC_ID_ACC = (MANUFAC_ID.compared.sum() / len(MANUFAC_ID) * 100)
    if len(MANUFAC_ID != 0):
        print('MANUFAC_ID', MANUFAC_ID_ACC)
        MANUFAC_ID_list.append(MANUFAC_ID_ACC)
    else:
        print('MANUFAC_ID Empty')
        MANUFAC_ID_list.append('MANUFAC_ID Empty')

    PROD_NAME = df[df['new_target'] == 'PROD_NAME']
    PROD_NAME_ACC = (PROD_NAME.compared.sum() / len(PROD_NAME) * 100)
    if len(PROD_NAME != 0):
        print('PROD_NAME', PROD_NAME_ACC)
        PROD_NAME_list.append(PROD_NAME_ACC)
    else:
        print('PROD_NAME Empty')
        PROD_NAME_list.append('PROD_NAME Empty')

    PROD_NUM = df[df['new_target'] == 'PROD_NUM']
    PROD_NUM_ACC = (PROD_NUM.compared.sum() / len(PROD_NUM)* 100)
    if len(PROD_NUM != 0):
        print('PROD_NUM', PROD_NUM_ACC)
        PROD_NUM_list.append(PROD_NUM_ACC)
    else:
        print('PROD_NUM Empty')
        PROD_NUM_list.append('PROD_NUM Empty')

    TITLE = df[df['new_target'] == 'TITLE']
    TITLE_ACC = (TITLE.compared.sum() / len(TITLE) * 100)
    if len(TITLE != 0):
        print('TITLE', TITLE_ACC)
        TITLE_list.append(TITLE_ACC)
    else:
        print('TITLE Empty')
        TITLE_list.append('TITLE Empty')

    META_DESCRIPTION = df[df['new_target'] == 'META_DESCRIPTION']
    META_DESCRIPTION_ACC = (META_DESCRIPTION.compared.sum() / len(META_DESCRIPTION) * 100)
    if len(META_DESCRIPTION != 0):
        print('META_DESCRIPTION', META_DESCRIPTION_ACC)
        META_DESCRIPTION_list.append(META_DESCRIPTION_ACC)
    else:
        print('META_DESCRIPTION Empty')
        META_DESCRIPTION_list.append('META_DESCRIPTION Empty')

csv_name = ['FILENAME']
list_files = os.listdir(PATH)
i = 0
for j in range(len(list_files)):
    print(i + len(list_files), 'of', (len(list_files)))
    dataset_filename = os.listdir(PATH)[j]
    dataset_path = os.path.join("../..", PATH, dataset_filename)
    print(dataset_filename)
    # BOOL_list.append(dataset_filename)
    # DATE_TIME_list.append(dataset_filename)
    # DATE_list.append(dataset_filename)
    # URL_list.append(dataset_filename)
    # PROD_PHOTO_URL_list.append(dataset_filename)
    # PDF_list.append(dataset_filename)
    # LANGUAGE_ID_list.append(dataset_filename)
    # COUNT_list.append(dataset_filename)
    # INTERNAL_ID_list.append(dataset_filename)
    # PROD_CAT_ID_list.append(dataset_filename)
    # PROD_TYPE_ID_list.append(dataset_filename)
    # PROD_WEIGHT_list.append(dataset_filename)
    # PRICE_list.append(dataset_filename)
    # CURRENCY_CODE_list.append(dataset_filename)
    # AUTHOR_list.append(dataset_filename)
    # DESC_LONG_list.append(dataset_filename)
    # MANUFAC_ID_list.append(dataset_filename)
    # PROD_NAME_list.append(dataset_filename)
    # PROD_NUM_list.append(dataset_filename)
    # TITLE_list.append(dataset_filename)
    # META_DESCRIPTION_list.append(dataset_filename)
    csv_name.append(dataset_filename)
    evaluate_classes_2(dataset_path)
    i -= 1
table = {'CSV': csv_name, 'BOOL': BOOL_list, 'DATE_TIME': DATE_TIME_list, 'DATE': DATE_list, 'URL': URL_list,
         'PROD_PHOTO_URL': PROD_PHOTO_URL_list, 'PDF': PDF_list, 'LANGUAGE_ID': LANGUAGE_ID_list,
         'COUNT': COUNT_list, 'INTERNAL_ID': INTERNAL_ID_list, 'PROD_CAT_ID': PROD_CAT_ID_list,
         'PROD_TYPE_ID': PROD_TYPE_ID_list, 'PROD_WEIGHT': PROD_WEIGHT_list, 'PRICE': PRICE_list,
         'CURRENCY_CODE': CURRENCY_CODE_list, 'AUTHOR': AUTHOR_list, 'DESC_LONG': DESC_LONG_list,
         'MANUFAC_ID': MANUFAC_ID_list, 'PROD_NAME': PROD_NAME_list, 'PROD_NUM': PROD_NUM_list, 'TITLE': TITLE_list,
         'META_DESCRIPTION': META_DESCRIPTION_list}
class_accuracies_df = pd.DataFrame(table)
os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\accuracies')
class_accuracies_df.to_csv('class_accuracies_no_Author.csv')

# class_accuracies = list(chain(BOOL_list, DATE_TIME_list, DATE_list, URL_list, PROD_PHOTO_URL_list, PDF_list,
#                                 LANGUAGE_ID_list, COUNT_list, INTERNAL_ID_list, PROD_CAT_ID_list, PROD_TYPE_ID_list,
#                                 PROD_WEIGHT_list, PRICE_list, CURRENCY_CODE_list, AUTHOR_list,
#                                 DESC_LONG_list, MANUFAC_ID_list, PROD_NAME_list, PROD_NUM_list, TITLE_list,
#                                 META_DESCRIPTION_list))
# print(class_accuracies)
# with open('class_accuracies.txt', 'w') as f:
#         os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\accuracies')
#         for item in class_accuracies:
#             f.write(f'{item}\n')

# my_list = ['foo', 'fob', 'faz', 'funk']
# string = 'bar'
# list2 = list(map(lambda orig_string: orig_string + string, my_list))