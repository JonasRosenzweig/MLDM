import os
import pickle
import keras
import json
import glob

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing import sequence


THRESHOLD = 0.51
SAMPLE_AMOUNT = 10
RANDOM_STATE = 7
TARGETS = ['Name', 'Description', 'Category', 'Price', 'Amount', 'EAN']

uploads_path = r'C:\Users\mail\Downloads\data\reformat_test\*.csv'
#uploads_path = r'/var/www/html/ADM/public/temp'

# list files in directory
list_files = glob.glob(uploads_path)

# select latest file in directory
recent_upload = max(list_files, key=os.path.getctime)

# load uploaded .csv into pandas dataframe
df = pd.read_csv(recent_upload,
                 error_bad_lines=False,
                 engine='c',
                 encoding='ISO-8859-1',
                 low_memory=False)

# sample SAMPLE_AMOUNT of rows from df with equal probability
if len(df.index >= SAMPLE_AMOUNT):
    try:
        df = df.sample(SAMPLE_AMOUNT,
                       random_state=RANDOM_STATE)
    except ValueError:
        pass
df = df.reset_index(drop=True)

# load NLP multi-class classification model file, tokenizer and label encoder
#model = keras.models.load_model(r'/var/www/html/ADM_Data/6_class_MVP.h5')
#tokenizer = pickle.load(open(r'/var/www/html/ADM_Data/6_class_MVP.pkl','rb'))
model = keras.models.load_model\
   (r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\organized\6_class_MVP.h5')
tokenizer = pickle.load\
    (open(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\organized\6_class_MVP.pkl', 'rb'))
le = LabelEncoder()

# function to make predictions
def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    predict_x = model.predict(text_pad)
    predict_class = np.argmax(predict_x, axis=1)
    score = le.inverse_transform(predict_class)
    prediction = score[0]
    return prediction

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
        class_predictions = []
        targets = TARGETS
        # fitting targets for label encoder
        targets = le.fit_transform(targets)
        data = df_select[m]
        predicted_class = predictClass(data, tokenizer, model)
        predictions_map[n].append(predicted_class)
        predictions_only[n].append(predicted_class)
        print('Data: {data}, Class Prediction: {prediction}'
              .format(data=data, prediction=predicted_class))

    predictions_only[n].pop(0)
    column_predictions = (predictions_only[n])
    column_predictions_series = pd.Series(column_predictions)
    column_predictions_count = column_predictions_series.value_counts()
    print('map list:', predictions_map)
    print('predictions only:', predictions_only)
    print('column predictions:', column_predictions)
    print('column predictions series:', column_predictions_series)
    print('column predictions count:', column_predictions_count)
    print('majority class:', column_predictions_count.index[0])
    print('number of predictions:', column_predictions_count.iloc[0])
    print('________________________')

    if column_predictions_count.iloc[0] > len(df_select) * THRESHOLD:
        print(column_predictions_count.index[0], 'is the Majority Predicted Class.')
        print('The majority class is: {pred}; {num_pred} of {len} predictions.'
              '\n Original Class: {origin}'
              .format(pred=column_predictions_count.index[0],
                      num_pred=column_predictions_count.iloc[0],
                      len=len(df_select),
                      origin=column_headers[n]))
        Predictions.append(column_predictions_count.index[0])
        Certainty.append(100)

    else:
        print('There is no Majority Predicted Class above the threshold.'
              '\n Original Class: {origin}'
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

# Transform lists into JSON serializable output
json_map = {"Columns": []}
json_pred_cert = {}

for i in range(len(column_headers)):
    json_map["Columns"].append({"Original Class {i}".format(i=i+1): column_headers[i],
                                "Model Prediction(s)": []})
    if isinstance(Predictions[i], list):
        json_map["Columns"][i]["Model Prediction(s)"] \
            .append({"Prediction": (Predictions[i]), "Certainty": (Certainty[i])})
    else:
        json_map["Columns"][i]["Model Prediction(s)"]\
            .append({"Prediction": [(Predictions[i])], "Certainty": [(Certainty[i])]})

json_map = json.dumps(json_map)

#os.chdir(r'/var/www/html/ADM_Data/json_maps')
os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\Maps')
filename = os.path.basename(recent_upload)
json_filename = filename.with_suffix('json')
with open(json_filename, 'w') as file:
    file.write(json_map)





