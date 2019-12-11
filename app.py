import numpy as np
from flask import Flask, request, jsonify, render_template ,send_file,make_response,Response
import joblib
import os
import time
import pandas as pd

app = Flask(__name__)

model = joblib.load('model.pkl')

y_columns = ['C5NP','C5IP','C5N','C6NP','C6IP','C6N','C6A','C7NP','C7IP','C7N','C7A',
'C8NP','C8IP','C8N','C8A','C9NP','C9IP','C9N','C9A','C10NP','C10IP','C10N','C10A']

# main page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/save',methods=['GET'])
def save():
	file_path = 'static/predict_result.xlsx'
	res = pd.DataFrame(result_dict,index=['value']).T
	res.to_excel(file_path)
	return send_file(file_path,as_attachment=True,cache_timeout=0)

@app.route('/predict',methods=['POST'])
def predict():
    features = np.array([float(x) for x in request.form.values()]).reshape(1, -1)
    preds = model.predict(features)[0]
    global result_dict
    result_dict = dict(zip(y_columns,[round(v,2) for v in preds]))
    return render_template('index.html',
    	prediction_text = 'result {}'.format(result_dict),
    	have_button = 'True')

if __name__ == "__main__":
    app.run(debug=True,port=5030)