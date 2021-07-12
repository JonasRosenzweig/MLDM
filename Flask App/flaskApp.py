from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        file = request.form['upload-file']
        data = pd.read_csv(file)
        print(data)
        return render_template('data.html', data=data.to_dict())



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=50000)
