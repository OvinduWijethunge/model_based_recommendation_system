# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:21:51 2021

@author: Acer
"""
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

# Load the model
filename = 'SVD_modelv2.pkl'
algo = pickle.load(open(filename, 'rb'))

df_rating = pd.read_csv('modified_rating_dataSet.csv')
df_movies = pd.read_csv('modified_movies_dataSet.csv')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def get_recommendations(uid):
    prediction_list = []
    
    for mid in df_rating['movieID'].unique():
        predic = algo.predict(uid,mid)
        prediction_list.append(predic)
    return prediction_list  


@app.route('/predict', methods=['GET','POST'])
def predict():
    
    
    if request.method == 'POST':
    
        uid = int(request.form['user_id'])
        #uid = 75
        is_user_availble = df_rating[df_rating['userID'] == uid].shape 
        if is_user_availble[0] == 0:
            text = ['This is a new user']
            return render_template('result.html', prediction=text)
        else:
            
            prediction = get_recommendations(uid)
            df_pred=pd.DataFrame(prediction)
            # Renaming our predictions to original names
            df_pred=df_pred.rename(columns={'uid':'userID', 'iid':'movieID','est':'rating'}).sort_values('rating',ascending=False).head(10)
            #print(df_pred['movieID'])
            df_final = df_pred.merge(df_movies ,on='movieID',how='left')
            movies_list = df_final['title'].values
            
            print(movies_list)
            return render_template('result.html', prediction=movies_list)

if __name__ == '__main__':
    app.run(debug=True)