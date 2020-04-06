import numpy as np
from flask import Flask, request, jsonify, render_template ,send_file,make_response,Response,flash,redirect,url_for
import joblib
import os
import time
import catboost
import pandas as pd
import datetime
from utils import *

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

model = joblib.load('./model.pkl')

# x_columns
x_columns = ['T10','T50','T90','N+A']

# y_columns
y_columns = ['C5NP','C5IP','C5N','C6NP','C6IP','C6N','C6A','C7NP','C7IP','C7N','C7A',
'C8NP','C8IP','C8N','C8A','C9NP','C9IP','C9N','C9A','C10NP','C10IP','C10N','C10A']

# main page
@app.route('/')
def home():
    return render_template('index.html')

# save method
@app.route('/save',methods=['GET'])
def save():
    # open excel
    file_path = 'static/predict_result.xlsx'
    table = pd.read_excel(file_path)
    # select Date, x_col and y_col
    if len(table) != 0:
        table = table[['Date']+x_columns+y_columns]
    # create new row
    row = pd.DataFrame(result_dict,index=[len(table)])
    # add now time as index
    row['Date'] = str(datetime.datetime.now())
    # sort columns
    row = row[['Date']+x_columns+y_columns]
    # append new row to excel
    table = table.append(row)
    # save excel
    table.to_excel(file_path,index=True)
    # send_file to user
    return send_file(file_path,as_attachment=True,cache_timeout=0)

# predict method
@app.route('/predict',methods=['POST'])
def predict():
    # get features
    features = np.array([float(x) for x in request.form.values()]).reshape(-1)
    X = pd.DataFrame(index=[0],columns=['T10','T50','T90','N+A','P'])
    X['T10'] = features[0]
    X['T50'] = features[1]
    X['T90'] = features[2]
    X['N+A'] = features[3]
    X['P'] = 100 - X['N+A']
    
    # check features
    c1 = (features[0]<features[1])&(features[1]<features[2])
    c2 = (features[0]>55)&(features[0]<116)
    c3 = (features[1]>84)&(features[1]<131)
    c4 = (features[2]>122)&(features[2]<198)
    c5 = (features[3]>30)&(features[3]<65)
    
    # check features
    error = ""
    if c1 != True:
        error += "錯誤！輸入溫度並非遞增\n"
    if c2 != True:
        error += "錯誤！T10範圍超出限制\n"
    if c3 != True:
        error += "錯誤！T50範圍超出限制\n"
    if c4 != True:
        error += "錯誤！T90範圍超出限制\n"
    if c5 != True:
        error += "錯誤！N+A範圍超出限制\n"
    
    # if feature error return error_msg
    if error != "":
        return render_template('index.html',error_msg = error)

    # predict y    
    preds = model.predict(X)[0]
    
    # get y result_dict
    global result_dict
    result_dict = dict(zip(y_columns,[round(v,4) for v in preds]))
    
    # add features to result_dict
    result_dict['T10'] = features[0]
    result_dict['T50'] = features[1]
    result_dict['T90'] = features[2]
    result_dict['N+A'] = features[3]
    
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
    
    # caculate total SUM
    df.loc[:,'SUM'] = df.sum(axis=1)
    df.loc['SUM',:] = df.sum(axis=0)
    
    # fill na
    df = df.fillna('--')
    
    # render html 
    return render_template('index.html',table = df.to_html(),have_button = 'True')

if __name__ == "__main__":
    app.run(debug = True,port = 5043)