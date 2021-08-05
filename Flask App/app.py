from flask import Flask, render_template, request
import pandas as pd
import csv

app = Flask(__name__)

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