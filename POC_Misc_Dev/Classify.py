import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
import keras
import re

SAVEPATH = r'C:\Users\mail\PycharmProjects\MLDM\Data\Classified Data'
CSV = r'C:\Users\mail\PycharmProjects\MLDM\Data\2D Data\vaiva.csv'

DATE_TIME_REGEX = '([0-9]|0[0-9]|1[0-9])-([0-9][0-9]|[0-9])-[0-9]{4} ([0-9]|0[0-9]|1[0-9])(.)[0-9]{2}:[0-9]{2}$'
DATE_REGEX = '^(0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])[- /.](19|20)\d\d$'
PRICE_REGEX = '^(\d{1,5})$|^(\d{1,5},\d{1,2})$|^(\d{1,2}\.\d{3,3})$|^(\d{1,2}\.\d{3,3},\d{1,2})$'
WEIGHT_REGEX = '^\d+\,\d\d\d\d$'
URL_REGEX = '[-a-zA-Z0-9@:%_\+.~#?&//=]{2,256}\.[a-z]{2,4}\b(\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?'
IMAGE_REGEX = '([0-9a-zA-Z\._-]+.(png|PNG|gif|GIF|jp[e]?g|JP[E]?G))'
INTEGER_REGEX = '(^\d{1,4}$)'

model_4 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\4ClassSimple.h5')
model_3 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\3ClassSimpleNumID.h5')
model_5 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\5ClassSimpleNumID.h5')
model_6 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\6ClassSimple.h5')
model_14 = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\14ClassSimple.h5')

print("Models Loaded")
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


def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    score = le.inverse_transform(model.predict_classes([text_pad]))
    return score[0]


def append_print(array, index, string):
    array.append(string)
    print(index, string)


def csvPredicter(csv):
    df = pd.read_csv(csv, names=['input', 'target'], encoding='ISO-8859-1', skiprows=1,
                     low_memory=False, index_col=False)
    df = df[~df.target.str.contains('|'.join(unused_headers))]
    df.astype(str)
    new_target = []
    for j in df['target']:
        if j in boolean:
            new_target.append('BOOL')
        elif j in date_time:
            new_target.append('DATE_TIME')
        elif j in date:
            new_target.append('DATE')
        elif j in url:
            new_target.append('URL')
        elif j in num_price:
            new_target.append('PRICE')
        # elif j in num_id_amt:
        #     new_target.append('NUMERIC_ID/AMT')
        elif j in author:
            new_target.append('AUTHOR')
        elif j in pdf:
            new_target.append('PDF')
        elif j in count:
            new_target.append('COUNT')
        else:
            new_target.append(j)
    pred = []
    for i in df['input']:
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
                append_print(pred, i,score)
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
            # score = 'UNKNOWN'
            targets = ['DESC_LONG', 'MANUFAC_ID', 'PROD_NAME', 'PROD_NUM', 'TITLE', 'META_DESCRIPTION']
            targets = le.fit_transform(targets)
            score = predictClass(i, tok_6, model_6)
            append_print(pred, i, score)
        table = {'input': df['input'],
                 'target': df['target'],
                 'pred': pred,
                 'new_target': new_target}
    dfClassed = pd.DataFrame(table)
    os.chdir(SAVEPATH)
    dfClassed.to_csv('testPredModels_vaiva.csv')


csvPredicter(CSV)

test_CSV = r'C:\Users\mail\PycharmProjects\MLDM\Data\Classified Data\testPredModels_vaiva.csv'
dfClassed = df = pd.read_csv(test_CSV, names=['input', 'target', 'pred', 'new_target'], encoding="ISO-8859-1",
                             skiprows=1,
                             low_memory=False, index_col=0)

compared = dfClassed['new_target'] == dfClassed['pred']
comp_table = {'input': dfClassed['input'],
              'target': dfClassed['target'],
              'pred': dfClassed['pred'],
              'new_target': dfClassed['new_target'],
              'compared': compared}
comp_df = pd.DataFrame(comp_table)

count_T = comp_df.compared.sum()
count_F = (len(comp_df) - (comp_df.compared.sum()))
count_B = (comp_df.pred == 'BOOL').sum()
count_U = (comp_df.pred == 'UNKNOWN').sum()

print(count_T, 'Correct Classifications out of', len(comp_df))
print(count_F, 'Incorrect Classifications out of', len(comp_df))
accuracy = (count_T / len(comp_df)) * 100
bool_percent = (count_B / len(comp_df)) * 100
unknown_percent = (count_U / len(comp_df)) * 100
print('Accuracy:', "{:.2f}".format(accuracy), '%')
print('Boolean:', "{:.2f}".format(bool_percent), '%')
print('Unknown:', "{:.2f}".format(unknown_percent), '%')
print('Semi-Classified Accuracy (total minus unknowns):', '{:.2f}'.format(100 - unknown_percent), '%')

# not classified with if statements (12): -- UNKNOWN
# PROD_NUM, PROD_NAME, PROD_SORT, PROD_SITE_SPECIFIC_SORTING, VENDOR_NUM, PROD_UNIT_ID, PROD_DELIVERY
# PROD_CREATED_BY, PROD_EDITED_BY, DESC_SHORT, DESC_LONG, DESC_LONG_2

# if statements/REGEX classified exact (5): -- EXACT
# LANGUAGE_ID, PROD_WEIGHT, PROD_PHOTO_URL, CURRENCY_CODE, DIRECT_LINK

# if statements/REGEX semi-classified (21): -- SEMI-CLASSIFIED
# NUMERIC_ID/AMT (5): PROD_MIN_BUY, PROD_MAX_BUY, PROD_TYPE_ID, INTERNAL_ID, PROD_CAT_ID
# PRICE (4): PROD_RETAIl_PRICE, PROD_COST,  UNIT_PRICE, PROD_COST_PRICE
# BOOL (7): PROD_HIDDEN, PROD_VAR_MASTER, PROD_NEW, PROD_FRONT_PAGE, TOPLIST_HIDDEN, OMIT_FROM_FREE_SHIPPING_LIMIT
# BOOL: PROD_SHOW_ON_GOOGLE_FEED
# DATE_TIME (2): PROD_CREATED, PROD_EDITED
# COUNT(4): PROD_SALES_COUNT, STOCK_COUNT, STOCK_LIMIT, PROD_VIEWED

# TODO: Look through prediction tests - f.x PROD_NUM and INTENRAL_ID for blossom
# TODO: add unknown prediction to predict method, based on confidence
# TODO: have predict show confidence of prediction