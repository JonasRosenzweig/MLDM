### Main Evaluate script - compares model predictons to known maps and outputs results ###

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
from os import listdir
# for path directory
from sklearn.preprocessing import LabelEncoder
# to encode and decode the target labels
from keras.preprocessing import sequence
# for sequence padding of data
from stdnum import ean
#  for EAN check validation

# read_csv method to change encoding params for global use
def read_csv(df):
    return pd.read_csv(df, error_bad_lines=False, engine='c', encoding='ISO-8859-14', low_memory=False, dtype=str)

# EAN Validator method
def validate_EAN(string):
    try:
        ean.validate(string)
        return True
    except:
        return False

# removes duplicates items from a list
def remove_duplicates(l):
    return list(dict.fromkeys(l))

# paths for model and tokenizer
Model_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Main\Trained Models\OTHER_NAME_PRICE_SIZE_COLOR_OVER_WORDEMB.h5'
Tokenizer_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Main\Trained Models\OTHER_NAME_PRICE_SIZE_COLOR_OVER_WORDEMB.pkl'

# header and target paths for evaluation
Mapped_Headers_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Manual maps\Headers.csv'
Mapped_Targets_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Manual maps\Targets.csv'

# load headers and targets into pandas dataframes
Mapped_Headers_df = read_csv(Mapped_Headers_path)
Mapped_Targets_df = read_csv(Mapped_Targets_path)

# lists of targets and headers for evaluation
Mapped_Targets_list = Mapped_Targets_df.values.tolist()
Mapped_Headers_list = Mapped_Headers_df.values.tolist()

# same lists, removing nan str values
Mapped_Targets_list = [[x for x in y if str(x) != 'nan'] for y in Mapped_Targets_list]
Mapped_Headers_list = [[x for x in y if str(x) != 'nan'] for y in Mapped_Headers_list]

# datafeed path
EVAL_path = r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Product Data feeds'
list_files = listdir(EVAL_path)

# targets list for model
TARGETS = ['OTHER', 'NAME', 'PRICE', 'SIZE', 'COLOR']

# Manual Map lists and features
OTHER_list, EAN_list, PRICE_list, NAME_list, COLOR_list, SIZE_list = [], [], [], [], [],[]
Features_lists = [OTHER_list, EAN_list, PRICE_list, NAME_list, COLOR_list, SIZE_list]

# lists for custom rules
OTHER_Headers = ['Washing','Ironing','Drying','Drycleaning','Bleaching','materiale','Materiale','%', 'Color Number',
                 'Type','type','Kolli','pct','Antal','Quantity','Qty','quantity','qty','Fit','lukning',
                 'Category','category','Kategori','kategori','Weight','weight','Qty.','Customer','Units','Age','age',
                 'delivery','date','Date','Delivery','description','Description','Delivery','Season','K??n','Gender',
                 'Account','No.','Stock','stock','pct.','month','Composition','composition','available','Finish',
                 'Order Closing','Washing','Beskrivelse 2', 'In Stock', 'Available','Date', 'Per Display', 'Weight',
                 'Division', 'Delivery', 'Brand', 'Drop', 'No', 'Quality', 'Weigth', 'length', 'height', 'Pieces',
                 'Style no', 'Colour no', 'Color no', 'Item no.', 'Article Number', 'Total number of pairs']
PRICE_Headers = ['price','Price','Cost','VAT','Retail','Subtotal','Sub total','total','sub total','subtotal','RRP',
                 'Wholesale','wholesale','udsalgspris','pris','Pris','Udsalgspris','Katalogpris','katalogpris']
COLOR_Headers = ['Colour name', 'Color name', 'Color Name', 'Colour Name', 'color name']
SIZE_Headers = ['Size', 'size', 'St??rrelse', 'st??rrelse', 'm??l', 'M??l', 'St?????rrelse']
Headers_Rule_List = [OTHER_Headers, PRICE_Headers, COLOR_Headers, SIZE_Headers]

# check if substring in list
def Substring_match(match_list, string):
    for y in range(len(match_list)):
        if match_list[y] in string:
            return True

# create list of lists (Headers_Rule_List) from individual header map lists
for i in range(len(Mapped_Targets_list)):
    for j in range(len(Mapped_Targets_list[i])):
        if (Mapped_Targets_list[i][j]) == 'OTHER':
            OTHER_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'EAN':
            EAN_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'PRICE':
            PRICE_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'NAME':
            NAME_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'COLOR':
            COLOR_list.append(Mapped_Headers_list[i][j])
        if (Mapped_Targets_list[i][j]) == 'SIZE':
            SIZE_list.append(Mapped_Headers_list[i][j])

# remove duplicates from features lists
for i in range(len(Features_lists)):
    Features_lists[i] = remove_duplicates(Features_lists[i])

model = keras.models.load_model(Model_path)
tokenizer = pickle.load(open(Tokenizer_path, 'rb'))
le = LabelEncoder()
print('Model, Tokenizer and LabelEncoder loaded.')



# class prediction function using trained keras model and tokenizer
# tokenizes text, performs prediction on it, trains label encoder, returns class prediction
def predictClass(text, tok, model):
    text_pad = sequence.pad_sequences(tok.texts_to_sequences([text]), maxlen=300)
    predict_x = model.predict(text_pad)
    predict_class = np.argmax(predict_x, axis=1)
    score = le.inverse_transform(predict_class)
    prediction = score[0]
    return prediction

Predictions_list = []

# evaluate method - takes a path of .csv product feed data
# returns evaluation metrics: .txt, .csv
# returns .json map
def evaluate(path):
    listfiles = listdir(path)
    # used for mapping entire column to the class whose percentage of mapped data is over the threshold
    THRESHOLD = 0.51
    # number of data points sampled for classification - 10 is used for testing, in practice at least 100 should be used
    SAMPLE_AMOUNT = 30
    # set random state for reproducibility
    RANDOM_STATE = 7

    # for file in path
    for i in range(len(listfiles)):
        Predictions = []
        correct = 0
        # correct counter for future automation of evaluation analysis
        dataset_filename = os.listdir(path)[i]
        print(dataset_filename)
        mapped_filename = 'mapped_' + dataset_filename
        dataset_path = os.path.join('../..', path, dataset_filename)
        df = pd.read_csv(dataset_path, error_bad_lines=False, engine='c', encoding='UTF-8', low_memory=False, dtype=str)
        # sample SAMPLE_AMOUNT of rows from df with equal probability
        if len(df.index) >= SAMPLE_AMOUNT:
            try:
                df = df.sample(SAMPLE_AMOUNT, random_state=RANDOM_STATE)
            except ValueError:
                pass
        df = df.reset_index(drop=True)
        column_headers = list(df.columns)
        # predictions map - predictions will be appended, matching the column header
        predictions_map = list(map(lambda item: [item], column_headers))
        # predictions only - map a list of size of column headers - headers will be popped
        # only includes predictions
        predictions_only = list(map(lambda item: [item], column_headers))

        # for column in file
        for j in range(len(column_headers)):
            df_select = df[column_headers[j]]
            df_select = df_select.astype(str)
            print('Original Class:', column_headers[j])

            # for data in column
            for k in range(len(df_select)):
                try:
                    print('Target:', Mapped_Targets_list[i][j+1])
                except IndexError:
                    pass
                targets = TARGETS
                targets = le.fit_transform(targets)
                data = df_select[k]
                print(data, column_headers[j])
                # Custom rules: substring matching
                if Substring_match(OTHER_Headers, column_headers[j]) or len(data) > 200:
                    predicted_class = 'CR-OTHER'
                    print('Custom rule applied - OTHER')
                elif Substring_match(PRICE_Headers, column_headers[j]):
                        predicted_class = 'CR-PRICE'
                        print('Custom rule applied - PRICE')
                elif Substring_match(COLOR_Headers, column_headers[j]):
                        predicted_class = 'CR-COLOR'
                        print('Custom rule applied - COLOR')
                elif Substring_match(SIZE_Headers, column_headers[j]):
                        predicted_class = 'CR-SIZE'
                        print('Custom rule applied - SIZE')
                elif validate_EAN(data):
                    predicted_class = 'EAN'
                elif data == 'nan':
                    predicted_class = 'NAN'
                else:
                    # model prediction
                    predicted_class = predictClass(data, tokenizer, model)
                    print('Model Prediction:', predicted_class)

                predictions_map[j].append(predicted_class)
                predictions_only[j].append(predicted_class)
            print('prediction:', predicted_class)
            print('_________')

            predictions_only[j].pop(0)
            # predictions_only is created from a dict with original headers, this pops them
            # leaving only the predictions
            column_predictions = (predictions_only[j])
            # predictions for selected column
            column_predictions_series = pd.Series(column_predictions)
            column_predictions_count = column_predictions_series.value_counts()
            # value counts to tally number of predictions for each prediction

            if column_predictions_count.iloc[0] > len(df_select) * THRESHOLD:
                Predictions.append(column_predictions_count.index[0])
            else:
                Predictions.append(column_predictions_count.index[0])
        Predictions_list.append(Predictions)
    os.chdir(r'C:\Users\mail\PycharmProjects\MLDM\Alpha\Organized Data\Output')
    with open('predictions_custom_rules_test.txt', 'w') as output:
        output.write(str(Predictions_list))
    predictions_df = pd.DataFrame(Predictions_list)
    predictions_df.to_csv('predictions_custom_rules_test.csv')

evaluate(EVAL_path)






