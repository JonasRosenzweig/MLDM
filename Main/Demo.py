
import re
import os
import pickle
import keras
import statistics

import numpy as np
import pandas as pd

from os import listdir
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence

COUNTRY_CODES = ['CN','CZ','FR','AT','JP','TR','DK','DE','HU','US','IE','GB','BA','PL','PT','IT','MT','IN','ES','PH',
'MG','TH','PK','AU','TW','SE','LT','VN','NL','CA','CD','MX','BE','DZ','BR','CH','ID','CL','KR','RO','LA','TN','FI','AF',
'AX','AL','AS','AD','AO','AI','AQ','AG','AR','AM','AW','AZ','BS','BH','BD','BB','BY','BZ','BJ','BM','BT','BO','BQ','BW',
'BV','IO','BN','BG','BF','BI','KH','CM','CV','KY','CF','TD','CX','CC','CO','KM','CG','CK','CR','CI','HR','CU','CW','CY',
'DJ','DM','DO','EC','EG','SV','GQ','ER','EE','ET','FK','FO','FJ','GF','PF','TF','GA','GM','GE','GH','GI','GR','GL','GD',
'GP','GU','GT','GG','GN','GW','GY','HT','HM','VA','HN','HK','IS','IR','IQ','IM','IL','JM','JE','JO','KZ','KE','KI','KP',
'KW','KG','LV','LB','LS','LR','LY','LI','LU','MO','MK','MW','MY','MV','ML','MH','MQ','MR','MU','YT','FM','MD','MC','MN',
'ME','MS','MA','MZ','MM','NA','NR','NP','NC','NZ','NI','NE','NG','NU','NF','MP','NO','OM','PW','PS','PA','PG','PY','PE',
'PN','PR','QA','RE','RU','RW','BL','SH','KN','LC','MF','PM','VC','WS','SM','ST','SA','SN','RS','SC','SL','SG','SX','SK',
'SI','SB','SO','ZA','GS','SS','LK','SD','SR','SJ','SZ','SY','TJ','TZ','TL','TG','TK','TO','TT','TM','TC','TV','UG','UA',
'AE','UM','UY','UZ','VU','VE','VG','VI','WF','EH','YE','ZM','ZW']

EAN_REGEX = '(?<=\s)\d{13}(?=\s)'


MODEL_DIR = \
    r'C:\Users\mail\PycharmProjects\MLDM\Data\Final\NAME_NUM_OTHER_Oversampled\models\NAME_NUM_OTHER_OVER_WORDEMB.h5'
TOKENIZER_DIR =\
    r'C:\Users\mail\PycharmProjects\MLDM\Data\Final\NAME_NUM_OTHER_Oversampled\models\NAME_NUM_OTHER_OVER_WORDEMB.pkl'

THRESHOLD = 0.51
SAMPLE_AMOUNT = 100
RANDOM_STATE = 7
TARGETS = ['PROD_NAME', 'PROD_NUM', 'OTHER']
model = keras.models.load_model(MODEL_DIR)
tokenizer = pickle.load(open(TOKENIZER_DIR, 'rb'))
le = LabelEncoder()
print('Model, Tokenizer and LabelEncoder loaded.')

def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    predict_x = model.predict(text_pad)
    predict_class = np.argmax(predict_x, axis=1)
    score = le.inverse_transform(predict_class)
    prediction = score[0]
    return prediction

PATH = r'C:\Users\mail\Downloads\data\Eval Datasets'
list_files = listdir(PATH)
k = 0


def evaluate(num):
    correct = 0
    dataset_filename = os.listdir(PATH)[num]
    print('________________________________________________________________________')
    print('----------{}----------'.format(dataset_filename))
    dataset_path = os.path.join("../..", PATH, dataset_filename)
    df = pd.read_csv(dataset_path, error_bad_lines=False, engine='c',
                 encoding='ISO-8859-14', low_memory=False, dtype=str)
    if len(df.index) >= SAMPLE_AMOUNT:
        try:
            df = df.sample(SAMPLE_AMOUNT,
                           random_state=RANDOM_STATE)
        except ValueError:
            pass
    df = df.reset_index(drop=True)
    column_headers = list(df.columns)
    # list of list of predictions including original column name
    predictions_map = list(map(lambda item: [item], column_headers))

    # list of list of predictions only - populate now then popped later
    predictions_only = list(map(lambda item: [item], column_headers))

    map_list = []
    maps_object = []
    Predictions = []
    Certainty = []


    for n in range(len(column_headers)):
        print('----------Mapping column {num} of {len}----------'
              .format(num=n+1, len=len(column_headers)))
        df_select = df[column_headers[n]]
        # sequence padding and tokenization of data requires string type
        df_select = df_select.astype(str)
        multiple_predictions = []
        multiple_certainties = []

        for m in range(len(df_select)):
            class_predictions =[]
            targets = TARGETS
            targets = le.fit_transform(targets)
            data = df_select[m]
            if bool(re.search(EAN_REGEX, data)) or len(data) == 13:
                predicted_class = 'PROD_BARCODE_NUM'
            elif len(data) > 200 or data == 'nan' or data in COUNTRY_CODES or len(data) < 5:
                predicted_class = 'OTHER'
            else:
                predicted_class = predictClass(data, tokenizer, model)
            #predictions_map[n].append(predicted_class)
            predictions_only[n].append(predicted_class)
            print('Data: {data}, Class Prediction: {prediction}'
                  .format(data=data, prediction=predicted_class))

        predictions_only[n].pop(0)
        column_predictions = (predictions_only[n])
        column_predictions_series = pd.Series(column_predictions)
        column_predictions_count = column_predictions_series.value_counts()
        print('----------Column Mapping Summary----------')
        #print('column predictions series:', column_predictions_series)
        #print('column predictions count:', column_predictions_count)
        print('predicted class:', column_predictions_count.index[0])
        print('number of classifications:', column_predictions_count.iloc[0])
        predictions_map[n].append(column_predictions_count.index[0])


        if column_predictions_count.iloc[0] > len(df_select) * THRESHOLD:
            print(column_predictions_count.index[0], 'is the Majority Predicted Class.')
            print('The majority class is: {pred}; {num_pred} of {len} predictions.'
                  '\nOriginal Class: {origin}'
                  .format(pred=column_predictions_count.index[0],
                          num_pred=column_predictions_count.iloc[0],
                          len=len(df_select),
                          origin=column_headers[n]))
            Predictions.append(column_predictions_count.index[0])
            Certainty.append(100)

        else:
            print('There is no Majority Predicted Class above the threshold.'
                  '\nOriginal Class: {origin}'
                  .format(origin=column_headers[n]))
            for i in range(len(column_predictions_count)):
                print('Predicted class {i}: {pred}; {num_pred} of {len} predictions.'
                      .format(i=i+1,
                              pred=column_predictions_count.index[i],
                              num_pred=column_predictions_count.iloc[i],
                              len=len(df_select)))
                multiple_predictions.append(column_predictions_count.index[i])
                multiple_certainties.append(column_predictions_count.iloc[i] / len(df_select) * 100)
            Predictions.append(multiple_predictions)
            Certainty.append(multiple_certainties)

        print('________________________________________________')

    print('----------Dataset Mapping Summary----------')
    print('map list:', predictions_map)
    #print('predictions only:', predictions_only)
    print('Predictions:', Predictions)
    print('Percentages:', Certainty)
    print(df.head())
    df_renamed = df.copy()
    print(len(Predictions))
    df_renamed.columns = Predictions
    del df_renamed['OTHER']
    print(df_renamed.head())
    Multi_Header1 = ['PRODUCTS']
    for y in range(len(df_renamed.columns)-1):
        Multi_Header1.append('')
    print('length cols', len(df_renamed.columns))
    print('length multi', len(Multi_Header1))
    df_renamed.columns = pd.MultiIndex.from_arrays([Multi_Header1, df_renamed.columns])
    print(df_renamed.head())



evaluate(21)