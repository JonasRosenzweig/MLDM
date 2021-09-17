import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
import keras
import re
from os import listdir
from sklearn.utils import shuffle

# Change Raw Data to 2D and reduce based on fraction
PATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\Raw Data'
CSV = 'vaiva.csv'
TEST = os.path.join(PATH, CSV)
frac = 0.01
print(TEST)
accuracies = []

def removePunct(name):
    name = name.replace(" ","_")
    name = name.replace(".", "_")
    return name


def cols_to_2D(df):
    columns = df.columns
    dfList = []
    for i in columns:
        x = df[i]
        x = pd.DataFrame(removePunct(x))
        x['Header'] = i
        x.rename(columns={i:'Data'}, inplace=True)
        dfList.append(x)
        y = pd.concat(dfList)
        y.dropna(axis=0, how='any', inplace=True)
    print(type(y))
    #print(y)
    return y

# Load requirements

# REGEX
DATE_TIME_REGEX = '([0-9]|0[0-9]|1[0-9])-([0-9][0-9]|[0-9])-[0-9]{4} ([0-9]|0[0-9]|1[0-9])(.)[0-9]{2}:[0-9]{2}$'
DATE_REGEX = '^(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.](19|20)\d\d$'
PRICE_REGEX = '^(\d{1,5})$|^(\d{1,5},\d{1,2})$|^(\d{1,2}\.\d{3,3})$|^(\d{1,2}\.\d{3,3},\d{1,2})$'
WEIGHT_REGEX = '^\d+\,\d\d\d\d$'
URL_REGEX = '[-a-zA-Z0-9@:%_\+.~#?&//=]{2,256}\.[a-z]{2,4}\b(\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?'
IMAGE_REGEX = '([0-9a-zA-Z\._-]+.(png|PNG|gif|GIF|jp[e]?g|JP[E]?G))'
INTEGER_REGEX = '(^\d{1,4}$)'

# Keras models
model_4 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\4ClassSimple.h5')
model_3 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\3ClassSimpleNumID.h5')
model_5 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\5ClassSimpleNumID.h5')
model_6 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\6ClassSimple.h5')
model_14 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\14ClassSimple.h5')
print("Models Loaded")

# Tokenizers
tokenizer_4 = open(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\tokenizer.pkl', 'rb')
tokenizer_3 = open(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\tokenizer_3_numID.pkl', 'rb')
tokenizer_5 = open(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\tokenizer_5_numID.pkl', 'rb')
tokenizer_6 = open(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\tokenizer_6.pkl', 'rb')
tokenizer_14 = open(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\tokenizer_14.pkl', 'rb')
print("Tokenizers loaded")

tok_3 = pickle.load(tokenizer_3)
tok_4 = pickle.load(tokenizer_4)
tok_5 = pickle.load(tokenizer_5)
tok_6 = pickle.load(tokenizer_6)
tok_14 = pickle.load(tokenizer_14)
le = LabelEncoder()
print("LabelEncoder Loaded")


# intermediate classes
num_id_amt = ['PROD_MIN_BUY', 'PROD_MAX_BUY', 'PROD_TYPE_ID', 'INTERNAL_ID', 'STOCK_COUNT', 'STOCK_LIMIT',
              'PROD_VIEWED', 'PROD_SALES_COUNT', 'PROD_CAT_ID', 'PROD_BARCODE_NUMBER']
num_price = ['PROD_RETAIl_PRICE', 'PROD_COST', 'UNIT_PRICE', 'PROD_COST_PRICE', 'PROD_RETAIL_PRICE']
boolean = ['PROD_HIDDEN', 'PROD_VAR_MASTER', 'PROD_NEW', 'PROD_FRONT_PAGE', 'TOPLIST_HIDDEN',
           'OMIT_FROM_FREE_SHIPPING_LIMIT', 'PROD_SHOW_ON_GOOGLE_FEED', 'PROD_SHOW_ON_GOOGLE_FEED',
           'PROD_SHOW_ON_FACEBOOK_FEED', 'PROD_SHOW_ON_PRICERUNNER_FEED', 'PROD_SHOW_ON_KELKOO_FEED']
date_time = ['PROD_CREATED', 'PROD_EDITED']
date = ['PROD_DELIVERY', 'PROD_DELIVERY_NOT_IN_STOCk']
pdf = ['PROD_PDF_URL', 'PROD_PDF_URL_2', 'PROD_PDF_URL_3', 'PROD_FILE_URL']
fields = ['FIELD_1', 'FIELD_2', 'FIELD_3', 'FIELD_4', 'FIELD_5', 'FIELD_6', 'FIELD_7', 'FIELD_8', 'FIELD_9', 'FIELD_10']
unsure = ['PROD_SORT', 'PROD_SITE_SPECIFIC_SORTING', 'PROD_UNIT_ID']
url = ['PROD_UNIQUE_URL_NAME', 'DIRECT_LINK']
count = ['PROD_SALES_COUNT', 'STOCK_COUNT', 'STOCK_LIMIT', 'PROD_VIEWED', 'PROD_MIN_BUY', 'PROD_MAX_BUY']
author = ['PROD_CREATED_BY', 'PROD_EDITED_BY']
unused_headers = ['PROD_PDF_URL_TEXT_3', 'PROD_PDF_URL_TEXT_2', 'PROD_PDF_URL_TEXT', 'PROD_HIDDEN_PERIOD_ID',
                  'PROD_FILE_URL', 'FIELD_8', 'PROD_NOTES', 'PROD_FRONT_PAGE_PERIOD_ID', 'FIELD_6',
                  'PROD_PDF_URL_2', 'PROD_PDF_URL', 'FIELD_3', 'PROD_UNIQUE_URL_NAME', 'FIELD_2', 'PROD_NEW_PERIOD_ID',
                  'FIELD_5', 'PROD_DELIVER_NOT_IN_STOCK', 'PROD_PICTURE_ALT_TEXT', 'PROD_GOOGLE_FEED_CATEGORY',
                  'PROD_LOCATION_NUMBER', 'DESC_LONG_2', 'DESC_SHORT', 'PROD_SEARCHWORD', 'VENDOR_NUM']


# Classifier


# Predict class of individual text
def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    score = le.inverse_transform(model.predict_classes([text_pad]))
    return score[0]


# helper function to print prediction
def append_print(array, index, string):
    array.append(string)
    print(index, string)


def classify(path, name):
    df = pd.read_csv(path, error_bad_lines=False, engine='c', encoding='ISO-8859-1',
                     low_memory=False, skiprows=1)
    df = df.sample(frac=frac, random_state=1)
    df = cols_to_2D(df)
    df = df.shuffle(df)
    print(df.head(100))
    df = df[~df.Header.str.contains('|'.join(unused_headers))]
    df = df.astype(str)

    intermediate_tar = []
    for j in df['Header']:
        if j in boolean:
            intermediate_tar.append('BOOL')
        elif j in date_time:
            intermediate_tar.append('DATE_TIME')
        elif j in date:
            intermediate_tar.append('DATE')
        elif j in url:
            intermediate_tar.append('URL')
        elif j in num_price:
            intermediate_tar.append('PRICE')
        elif j in author:
            intermediate_tar.append('AUTHOR')
        elif j in pdf:
            intermediate_tar.append('PDF')
        elif j in count:
            intermediate_tar.append('COUNT')
        else:
            intermediate_tar.append(j)

    pred = []
    for i in df['Data']:
        if i == 'True' or i == 'False':
            score = 'BOOL'
            append_print(pred, i, score)
        elif re.search(DATE_TIME_REGEX, i):
            score = 'DATE_TIME'
            append_print(pred, i, score)
        elif re.search(DATE_REGEX, i):
            score = 'DATE'
            append_print(pred, i, score)
        elif re.search(URL_REGEX, i) or '.html' in i or 'https://' in i:
            score = 'URL'
            append_print(pred, i, score)
        elif re.search(IMAGE_REGEX, i):
            score = 'PROD_PHOTO_URL'
            append_print(pred, i, score)
        elif '.pdf' in i:
            score = 'PDF'
            append_print(pred, i, score)
        elif i == '26':
            score = 'LANGUAGE_ID'
            append_print(pred, i, score)
        elif i.isnumeric():
            if re.search(INTEGER_REGEX, i):
                score = 'COUNT'
                append_print(pred, i, score)
            else:
                targets_num = ['INTERNAL_ID', 'PROD_CAT_ID', 'PROD_TYPE_ID']
                targets_num = le.fit_transform(targets_num)
                score = predictClass(i, tok_3, model_3)
                append_print(pred, i, score)
        elif re.search(WEIGHT_REGEX, i):
            score = 'PROD_WEIGHT'
            append_print(pred, i, score)
        elif re.search(PRICE_REGEX, i):
            score = 'PRICE'
            append_print(pred, i, score)
        elif i == 'DKK' or i == 'SEK' or i == 'EUR':
            score = 'CURRENCY_CODE'
            append_print(pred, i, score)
        elif i == 'admin' or 'mk' or ' sk' or 'jr' or 'pv':
            score = 'AUTHOR'
            append_print(pred, i, score)
        else:
            targets = ['DESC_LONG', 'MANUFAC_ID', 'PROD_NAME', 'PROD_NUM', 'TITLE', 'META_DESCRIPTION']
            targets = le.fit_transform(targets)
            score = predictClass(i, tok_6, model_6)
            append_print(pred, i, score)
        table = {'input': df['Data'],
                 'target': df['Header'],
                 'pred': pred,
                 'intermediate_tar': intermediate_tar}

    df_intermediate = pd.DataFrame(table)
    print(df_intermediate)
    compared = df_intermediate['intermediate_tar'] == df_intermediate['pred']
    compared_table = {'input': df_intermediate['input'],
                      'target': df_intermediate['target'],
                      'pred': df_intermediate['pred'],
                      'new_target': df_intermediate['intermediate_tar'],
                      'compared': compared}
    compared_df = pd.DataFrame(compared_table)
    count_T = compared_df.compared.sum()
    count_F = len(compared_df) - count_T
    print(count_T, 'Correct Classifications out of', len(compared_df))
    print(count_F, 'Incorrect Classifications out of', len(compared_df))
    accuracy = (count_T / len(compared_df)) * 100
    print('Accuracy:', "{:.2f}".format(accuracy), '%')
    accuracies.append(accuracy)
    os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\test')
    compared_df.to_csv(name, index=False)


list_files = listdir(PATH)
print("Starting transform of all files in", PATH)
i = 0
for j in range(len(list_files)):
    print(i + len(list_files),  'of', (len(list_files)), "files remaining.")
    dataset_filename = os.listdir(PATH)[j]
    dataset_path = os.path.join("../..", PATH, dataset_filename)
    classify(dataset_path, dataset_filename)
    i -= 1

with open('accuracies_001.txt', 'w') as f:
    os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\accuracies')
    for item in accuracies:
        f.write(f'{item}\n')
