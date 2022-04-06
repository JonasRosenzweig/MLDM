### Current Column Mapper script deployed to ADM server ###
# ADM paths commented out and replaced with local machine paths

import os
# os functions - save directory, recent file lookup
import pickle
# to load the tokenizer
import keras
#  for model loading
import numpy as np
# for argmax call in predictClass method
import pandas as pd
# for loading the data feed .csv into a pandas DataFrame
from sklearn.preprocessing import LabelEncoder
# to encode and decode the target labels
from keras.preprocessing import sequence
# for sequence padding of data
import json
# for json output
from pathlib import Path
import glob
# for path directory

# Constants
threshold = 0.51
# used for mapping entire column to the class whose percentage of mapped data is over the threshold
sample_amount = 10
# number of data points sampled for classification - 10 is used for testing, in practice at least 100 should be used


PATH2 = r'C:\Users\mail\Downloads\data\test\*.csv'
list_files2 = glob.glob(PATH2)
CSV2 = max(list_files2, key=os.path.getctime)
print('CSV2', CSV2)
print('CSV2File', os.path.basename(CSV2))

#CSV = r'/var/www/html/ADM_Data/Joha produkter.csv'
#CSV = r'C:\Users\mail\Downloads\data\Joha produkter.csv'
#PATH = r'/var/www/html/ADM/public/temp'
PATH = r'C:\Users\mail\Downloads\data\test'
list_files = os.listdir(PATH)
#CSV = r'C:\Users\mail\Downloads\data\Modern_Classic_Opstart.csv'
for j in range(len(list_files)):
    dataset_filename = os.listdir(PATH)[j]
    CSV = os.path.join("../..", PATH, dataset_filename)
print(dataset_filename)

df = pd.read_csv(CSV, error_bad_lines=False, engine='c', encoding='ISO-8859-1', low_memory=False)
df = df.astype(str)
#print(len(df.index))
#print(len(df.index) > sample_amount)
if len(df.index > sample_amount):
    try:
        df = df.sample(sample_amount, random_state=7)
    except ValueError:
        pass
df = df.reset_index(drop=True)
#model = keras.models.load_model(r'/var/www/html/ADM_Data/6_class_MVP.h5')
#tokenizer = pickle.load(open(r'/var/www/html/ADM_Data/6_class_MVP.pkl','rb'))
model = keras.models.load_model(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\organized\6_class_MVP.h5')
tokenizer = pickle.load(open(r'C:\Users\mail\PycharmProjects\MLDM\Main\Models\organized\6_class_MVP.pkl', 'rb'))
le = LabelEncoder()
print('Model, Tokenizer and LabelEncoder loaded.')


def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    predict_x = model.predict(text_pad)
    predict_class = np.argmax(predict_x, axis=1)
    score = le.inverse_transform(predict_class)
    return score[0]


col_dict = {df.columns.get_loc(c): c for idx, c in enumerate(df.columns)}
map_list = []
maps_object = []
Predictions = []
Certainty = []

lists = list(map(list, col_dict.items()))
Columns = list(df.columns)

for n in (range(len(col_dict))):
    print('----------Mapping column {num} of {len}----------'.format(num=n, len=len(col_dict)))
    df_select = df[col_dict[n]]
    df_select = df_select.astype(str)
    class_map = []
    multiple_pred = []
    multiple_predictions = []
    multiple_certainties = []
    for m in (range(len(df_select))):
        class_predictions = []
        targets = ['Name', 'Description', 'Category', 'Price', 'Amount', 'EAN']
        targets = le.fit_transform(targets)
        text = df_select[m]
        predicted_class = predictClass(text, tokenizer, model)
        lists[n].append(predicted_class)
        print('data:', text, ', prediction:', predicted_class)
    predicted_classes = pd.Series(lists[n])
    predicted_count = pd.Series(predicted_classes.value_counts())
    print('_____________________________')
    if predicted_count.iloc[0] > (len(df_select)) * threshold:
        print(predicted_count.index[0], 'is the Majority Predicted Class.')
        print('Majority Predicted classes:', predicted_count.index[0], ', predictions:', predicted_count.iloc[0],
              '/', len(df_select))
        class_predictions.append('Pred: {class_pred}: {num_pred} / {class_len}. Original: {origin}'
                                 .format(class_pred=predicted_count.index[0], num_pred=predicted_count.iloc[0],
                                         class_len=len(df_select),
                                         origin=col_dict[n]))
        class_map.append([col_dict[n], predicted_count.index[0]])
        Predictions.append(predicted_count.index[0])
        Certainty.append(100)


    else:
        print(predicted_count)
        print('There is no Majority Predicted Class above the threshold.')
        for i in range(len(predicted_count) - 2):
            print('Predicted classes:', predicted_count.index[i], ', predictions:', predicted_count.iloc[i], '/',
                  len(df_select))
            class_predictions.append(
                'Pred {num}: {class_pred}: {num_pred} / {class_len}. Original: {origin}'
                    .format(num=(i + 1), class_pred=predicted_count.index[i], num_pred=predicted_count.iloc[i],
                            class_len=len(df_select),
                            origin=col_dict[n]))
            multiple_pred.append([col_dict[n], predicted_count.index[i]])
            multiple_predictions.append(predicted_count.index[i])
            multiple_certainties.append(predicted_count.iloc[i]/len(df_select)*100)
        class_map.append(multiple_pred)
        Predictions.append(multiple_predictions)
        Certainty.append(multiple_certainties)
    print('Original Class:', col_dict[n])
    print('__________________________________________')
    map_list.append(class_predictions)
    maps_object.append(class_map)

print('Predictions:', Predictions)
print('Certainty:', Certainty)

json_map = {"Columns": []}
json_pred_cert = {}
print(json_map)

for i in range(len(Columns)):
    json_map["Columns"].append({"Original Class {i}".format(i=i+1): Columns[i],
                                "Model Prediction(s)": []})
    if isinstance(Predictions[i], list):
        json_map["Columns"][i]["Model Prediction(s)"] \
            .append({"Prediction": (Predictions[i]), "Certainty": (Certainty[i])})
    else:
        json_map["Columns"][i]["Model Prediction(s)"]\
            .append({"Prediction": [(Predictions[i])], "Certainty": [(Certainty[i])]})

json_map = json.dumps(json_map)

#os.chdir(r'/var/www/html/ADM_Data/json_maps')

dataset_filename = Path(os.path.basename(CSV2))
dataset_filename = dataset_filename.with_suffix('.json')
print('file', dataset_filename)
os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Data\Maps')
with open(dataset_filename, 'w') as file:
    file.write(json_map)

