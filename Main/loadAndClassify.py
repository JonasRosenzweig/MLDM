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
        elif '.jpg' in i or '.jpeg' in i:
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
            append_print(pred, i ,score)
        else:
            score = 'UNKNOWN'
            # score = predictClass(i)
            append_print(pred, i, score)
    table = {'input': df['input'],
             'target': df['target'],
             'pred': pred}
    dfPred = pd.DataFrame(table)
    os.chdir(SAVEPATH)
    dfPred.to_csv('testPredict.csv')


csvPredicter(CSV)

# not classified: PROD_NUM, PROD_NAME
