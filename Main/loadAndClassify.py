import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence
import keras
import re

SAVEPATH = r'C:\Users\surface\Desktop\YouWe\MLDM\Data\Classified Data'
CSV = r'C:\Users\surface\Desktop\YouWe\MLDM\Data\2D Data\vaiva.csv'

DATE_TIME_REGEX = '([0-9]|0[0-9]|1[0-9])-([0-9][0-9]|[0-9])-[0-9]{4} ([0-9]|0[0-9]|1[0-9])(.)[0-9]{2}:[0-9]{2}$'
PRICE_REGEX = '^(\d{1,5})$|^(\d{1,5},\d{1,2})$|^(\d{1,2}\.\d{3,3})$|^(\d{1,2}\.\d{3,3},\d{1,2})$'
WEIGHT_REGEX = '^\d+\,\d\d\d\d$'

model = keras.models.load_model(r'C:\Users\surface\Desktop\YouWe\MLDM\Main\Models\4ClassSimple.h5')
print("Model Loaded")
tokenizer = open(r'C:\Users\surface\Desktop\YouWe\MLDM\Main\Models\tokenizer.pkl', 'rb')
print("Tokenizer loaded")
tok = pickle.load(tokenizer)
le = LabelEncoder()
print("LabelEncoder Loaded")


def predictClass(text):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    score = le.inverse_transform(model.predict_classes([text_pad]))
    return score[0]


def append_print(array, index, string):
    array.append(string)
    print(index, string)


def csvPredicter(csv):
    df = pd.read_csv(csv, names=['input', 'target'], encoding="ISO-8859-1", skiprows=1,
                     low_memory=False, index_col=False)
    df.astype(str)
    targets = df.target
    targets = le.fit_transform(targets)
    print("LabelEncoder Fitted")
    pred = []
    for i in df['input']:
        if i == 'True' or i == 'False':
            score = 'BOOL'
            append_print(pred, i, score)
        elif i == 'DKK' or i == 'SEK' or i == 'EUR':
            score = 'CURRENCY_CODE'
            append_print(pred, i, score)
        elif i == '26':
            score = 'LANGUAGE_ID'
            append_print(pred, i, score)
        elif '.jpg' in i or '.jpeg' in i or '.png' in i:
            score = 'PROD_PHOTO_URL'
            append_print(pred, i, score)
        elif '.html' in i or 'https://' in i:
            score = 'DIRECT_LINK'
            append_print(pred, i, score)
        elif i.isnumeric():
            score = 'NUMERIC_ID/AMT'
            append_print(pred, i, score)
        elif re.search(DATE_TIME_REGEX, i):
            score = 'DATE_TIME'
            append_print(pred, i, score)
        elif re.search(PRICE_REGEX, i):
            score = 'PRICE'
            append_print(pred, i, score)
        elif re.search(WEIGHT_REGEX, i):
            score = 'PROD_WEIGHT'
            append_print(pred, i, score)
        else:
            score = 'UNKNOWN'
            # score = predictClass(i)
            append_print(pred, i, score)
    table = {'input': df['input'],
             'target': df['target'],
             'pred': pred}
    dfPred = pd.DataFrame(table)
    os.chdir(SAVEPATH)
    dfPred.to_csv('testPredict_vaiva.csv')


csvPredicter(CSV)

test_CSV = r'C:\Users\surface\Desktop\YouWe\MLDM\Data\Classified Data\testPredict_vaiva.csv'
comp_df = df = pd.read_csv(test_CSV, names=['input', 'target', 'pred'], encoding="ISO-8859-1", skiprows=1,
                           low_memory=False, index_col=0)
compared = comp_df['target'] == comp_df['pred']
table = {'input': df['input'],
         'target': df['target'],
         'pred': df['pred'],
         'compared': compared}
comp_df = pd.DataFrame(table)
os.chdir(SAVEPATH)
comp_df.to_csv('testPredictCompared_vaiva.csv')

count_T = comp_df.compared.sum()
count_F = (len(comp_df) - (comp_df.compared.sum()))
count_B = (comp_df.pred == 'BOOL').sum()
count_U = (comp_df.pred == 'UNKNOWN').sum()

print(count_T, 'Correct Classifications out of', len(comp_df))
print(count_F, 'Incorrect Classifications out of', len(comp_df))
accuracy = (count_T/len(comp_df))*100
bool_percent = (count_B/len(comp_df))*100
unknown_percent = (count_U/len(comp_df))*100
print('Accuracy:', "{:.2f}".format(accuracy), '%')
print('Boolean:', "{:.2f}".format(bool_percent), '%')
print('Unknown:', "{:.2f}".format(unknown_percent), '%')
print('Semi-Classified Accuracy with Booleans:', '{:.2f}'.format(accuracy+bool_percent), '%')
print('Semi-Classified Accuracy (total minus unknowns):', '{:.2f}'.format(100-unknown_percent), '%')

# not classified with if statements (12): -- UNKNOWN
# PROD_NUM, PROD_NAME, PROD_SORT, PROD_SITE_SPECIFIC_SORTING, VENDOR_NUM, PROD_UNIT_ID, PROD_DELIVERY
# PROD_CREATED_BY, PROD_EDITED_BY, DESC_SHORT, DESC_LONG, DESC_LONG_2

# if statements/REGEX classified exact (5): -- EXACT
# LANGUAGE_ID, PROD_WEIGHT, PROD_PHOTO_URL, CURRENCY_CODE, DIRECT_LINK

# if statements/REGEX semi-classified (21): -- SEMI-CLASSIFIED
# NUMERIC_ID/AMT (9): PROD_MIN_BUY, PROD_MAX_BUY, PROD_TYPE_ID, INTERNAL_ID, STOCK_COUNT, STOCK_LIMIT, PROD_VIEWED
# NUMERIC_ID/AMT: PROD_SALES_COUNT, PROD_CAT_ID
# PRICE (3): PROD_RETAIl_PRICE, PROD_COST,  UNIT_PRICE
# BOOL (7): PROD_HIDDEN, PROD_VAR_MASTER, PROD_NEW, PROD_FRONT_PAGE, TOPLIST_HIDDEN, OMIT_FROM_FREE_SHIPPING_LIMIT
# BOOL: PROD_SHOW_ON_GOOGLE_FEED
# DATE_TIME (2): PROD_CREATED, PROD_EDITED

# 5th commit, only regex no models: 13.65% accuracy on vaiva.csv, 10.82% on milkywalk.csv
# 6th commit, checking with booleans: 35.49% on vaiva.csv (21.84% bool), 28.15% on milkywalk (17.33% bool)
# 7th commit: checking % unknowns: 73.83% vaiva (35.49% unknown), 64.62% milywalk (35.38% unknown)
