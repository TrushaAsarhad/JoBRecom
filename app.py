import streamlit as st
from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

# Load your models
tfidf = load('tfidf_vectorizer.pkl')
cosine_sim = load('cosine_similarity.pkl')
data = pd.read_csv('job_data.csv')

app = Flask(__name__)

# Define your job recommendation function
def get_recommendations(job_title, cosine_sim=cosine_sim):
    # Get the index of the job that matches the title
    idx = data[data['Job Title'] == job_title].index[0]

    # Get the pairwise similarity scores of all jobs with the given job
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the jobs based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 10 most similar jobs
    job_indices = [i[0] for i in sim_scores[1:11]]

    # Return the top 10 most similar jobs
    return data[['Job Title', 'Company Name', 'Location', 'skills']].iloc[job_indices]

# Flask route to recommend jobs
@app.route('/recommend', methods=['POST'])
def recommend():
    job_title = request.json.get('job_title')
    recommendations = get_recommendations(job_title)
    return jsonify(recommendations.to_dict())

# Home route
@app.route('/')
def home():
    return "Flask app is running!"

if __name__ == '__main__':
    app.run(port=5000)
