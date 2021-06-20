# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 18:21:51 2021

@author: Acer
"""
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# Load the model
filename = 'SVD_modelv2.pkl'
algo = pickle.load(open(filename, 'rb'))

df_rating = pd.read_csv('modified_rating_dataSet.csv')
df_movies = pd.read_csv('modified_movies_dataSet.csv')

dff = pd.read_csv('modified_content_dataSet.csv')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')
    
    
    
@app.route('/content_based')
def content_based():
    return render_template('content_based.html')    
    
    
    
@app.route('/colab_based')
def colab_based():
    return render_template('colab_based.html')        
    
    
    

def get_recommendations(uid):
    prediction_list = []
    
    for mid in df_rating['movieID'].unique():
        predic = algo.predict(uid,mid)
        prediction_list.append(predic)
    return prediction_list  


@app.route('/colab_predict', methods=['GET','POST'])
def colab_predict():
    
    
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
            
 


'''
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(sig[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices].values.tolist()[:10]
'''


 
@app.route('/content_predict', methods=['GET','POST'])
def content_predict(): 

    if request.method == 'POST':
    
        name = str(request.form['Movie name'])
        df = dff
        tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(df['description']) 
        sig = linear_kernel(tfidf_matrix, tfidf_matrix)  
        df = df.reset_index()
        titles = df['title']
        indices = pd.Series(df.index, index=df['title']) 
        
        idx = indices[name]
        sim_scores = list(enumerate(sig[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        movie_indices = [i[0] for i in sim_scores]
        movies_list = titles.iloc[movie_indices].values.tolist()[:10]
        #movies_list = get_recommendations(name).head(10)        
        
        print(movies_list)
        return render_template('result.html', prediction=movies_list)












           

if __name__ == '__main__':
    app.run(debug=True)