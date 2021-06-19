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
filename = 'SVD_model.pkl'
algo = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    
    df_new = pd.read_csv('colab_dataSet.csv')
    df_movies = pd.read_table("dataSet/movies.dat",sep="\t")
    if request.method == 'POST':
    
        #uid = int(request.form['user_id'])
        uid = 1123
        prediction_list = []
        for mid in df_new['movieID'].unique():
            predic = algo.predict(uid,mid) 
            prediction_list.append(predic)
        dfs_pred=pd.DataFrame(prediction_list)
        dfs_pred.drop(columns=['details'],inplace=True)
        dfs_pred=dfs_pred.rename(columns={'uid':'userID', 'iid':'id','est':'rating'})
        dfs_pred = dfs_pred.sort_values('rating',ascending=False)
        df_1 = dfs_pred.head(10)
        df2 = df_1.merge(df_movies ,on='id',how='left')
        recom_movies_list = df2[['title']].values
        print(recom_movies_list)
        #return recom_movies_list
        #recom_movies_list = df_1[['id']].values
        return render_template('result.html', prediction=recom_movies_list)

if __name__ == '__main__':
    app.run(debug=True)