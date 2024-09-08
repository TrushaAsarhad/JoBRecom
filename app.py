import streamlit as st
from joblib import load
import pandas as pd

# Load your models and data
tfidf = load('tfidf_vectorizer.pkl')
cosine_sim = load('cosine_similarity.pkl')
data = pd.read_csv('job_data.csv')

# Define the job recommendation function
def get_recommendations(job_title, cosine_sim=cosine_sim):
    if job_title not in data['Job Title'].values:
        return pd.DataFrame(columns=['Job Title', 'Company Name', 'Location', 'skills'])

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

# Streamlit UI
st.title("Job Recommendation System")

# Input for job title
job_title = st.text_input("Enter Job Title")

# Button to get recommendations
if st.button("Get Recommendations"):
    if job_title:
        recommendations = get_recommendations(job_title)
        if not recommendations.empty:
            st.write(recommendations)
        else:
            st.write("No recommendations found. Please try another job title.")
    else:
        st.error("Please enter a job title.")
