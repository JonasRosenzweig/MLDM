import flask
from flask import Flask, render_template, request
import pandas as pd
import csv
import tensorflow as tf
import keras
from keras.models import load_model

app = Flask(__name__)

# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     keras.backend.get_session().run(tf.local_variables_initializer())
#     return auc


global graph
graph = tf.compat.v1.get_default_graph()
model = load_model('4ClassSimple.h5')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    data = {'success': False}
    params = flask.request.json
    if (params == None):
        params = flask.request.args
    if (params != None):
        x = pd.DataFrame.from_dict(params, orient='index').transpose()
        with graph.as_default():
            data['prediction'] = str(model.predict(x)[0][0])
            data['success'] = True
        return flask.jsonify(data)


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        f = request.form['csvfile']
        data = []
        with open(f) as file:
            csvfile = csv.reader(file)
            for row in csvfile:
                data.append(row)
            data = pd.DataFrame(data)
            return render_template('data.html', data=data.to_html(header=False))


if __name__ == '__main__':
    app.run(debug=True)
