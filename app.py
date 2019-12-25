import numpy as np
from flask import Flask, request, jsonify, render_template ,send_file,make_response,Response,flash,redirect,url_for
import joblib
import os
import time
import pandas as pd
import datetime

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# model
model = joblib.load('model.pkl')

# y_columns
y_columns = ['C5NP','C5IP','C5N','C6NP','C6IP','C6N','C6A','C7NP','C7IP','C7N','C7A',
'C8NP','C8IP','C8N','C8A','C9NP','C9IP','C9N','C9A','C10NP','C10IP','C10N','C10A']

# main page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/save',methods=['GET'])
def save():
    # open excel
    file_path = 'static/predict_result.xlsx'
    table = pd.read_excel(file_path)
    # create new row
    row = pd.DataFrame(result_dict,index=[len(table)])
    row['Date'] = str(datetime.datetime.now())
    row = row[['Date']+y_columns]
    # append row to excel
    table = table.append(row)
    table.to_excel(file_path,index=True)
    return send_file(file_path,as_attachment=True,cache_timeout=0)

@app.route('/predict',methods=['POST'])
def predict():

    # get x 
    features = np.array([float(x) for x in request.form.values()]).reshape(1, -1)
    
    # check features
    c1 = (features[0][0]<features[0][1])&(features[0][1]<features[0][2])
    c2 = (features[0][0]>55)&(features[0][0]<116)
    c3 = (features[0][1]>84)&(features[0][1]<131)
    c4 = (features[0][2]>122)&(features[0][2]<198)
    c5 = (features[0][3]>30)&(features[0][3]<65)
    is_pass = c1&c2&c3&c4&c5
    
    if is_pass != True:
        error = '輸入特徵有問題,麻煩重新確認後再輸入一次'
        return render_template('index.html',error_msg = error)

    # predicy y    
    preds = model.predict(features)[0]
    
    # get result_dict
    global result_dict
    result_dict = dict(zip(y_columns,[round(v,2) for v in preds]))
    
    # get result dataframe
    df = pd.DataFrame(index=['C5','C6','C7','C8','C9','C10'],columns=['NP','IP','N','A'])
    for y_name in y_columns:
        if 'C10' not in y_name:
            idx = y_name[:2]
            col = y_name[2:]
        if 'C10' in y_name:
            idx = y_name[:3]
            col = y_name[3:]
        df.loc[idx,col] = result_dict[y_name]
    df = df.fillna('--')
    
    # render html 
    return render_template('index.html',table = df.to_html(),have_button = 'True')

if __name__ == "__main__":
    app.run(debug = True,port = 5042)