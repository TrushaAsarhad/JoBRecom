import streamlit as st
from joblib import load
import pandas as pd

# Load your models
tfidf = load('tfidf_vectorizer.pkl')
cosine_sim = load('cosine_similarity.pkl')
data = pd.read_csv('job_data.csv')

# Define your job recommendation function
def get_recommendations(job_title, cosine_sim=cosine_sim):
  # ... (your recommendation logic here)
  return data[['Job Title', 'Company Name', 'Location', 'skills']].iloc[job_indices]

# Streamlit App
st.title("Job Recommendation Tool")

job_title = st.text_input("Enter a Job Title:", key="job_title")

# Display a progress bar while fetching recommendations
with st.spinner("Finding similar jobs..."):
  if st.button("Recommend Similar Jobs"):
    recommendations = get_recommendations(job_title)
    st.success("Done!")

# Display recommendations or error message
if recommendations.empty:  # Now `recommendations` should be defined
    st.write("No similar jobs found!")
else:
    st.table(recommendations)