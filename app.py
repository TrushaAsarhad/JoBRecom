import streamlit as st
from joblib import load
import pandas as pd

# Load your models
tfidf = load('tfidf_vectorizer.pkl')
cosine_sim = load('cosine_similarity.pkl')
data = pd.read_csv('job_data.csv')

# Define your job recommendation function
def get_recommendations(job_title, cosine_sim=cosine_sim):
  """
  This function retrieves the top 10 most similar jobs based on job title.
  Args:
      job_title (str): The job title to find recommendations for.
      cosine_sim (sparse matrix): The pre-computed cosine similarity matrix.
  Returns:
      pandas.DataFrame: A DataFrame containing the top 10 recommended jobs.
  """
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

# Streamlit App
st.title("Job Recommendation Tool")

job_title = st.text_input("Enter a Job Title:", key="job_title")

# Display a progress bar while fetching recommendations
with st.spinner("Finding similar jobs..."):
  if st.button("Recommend Similar Jobs"):
    recommendations = get_recommendations(job_title)
    st.success("Done!")

# Display recommendations or error message
if recommendations.empty:
    st.write("No similar jobs found!")
else:
    st.table(recommendations)