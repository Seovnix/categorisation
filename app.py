import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load the SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu', cache_folder='./')

model = load_model()

# Streamlit UI
st.title("Keyword Category Classifier")
st.write("Upload your candidate labels file (CSV) and a `keywords.txt` file. The app will classify the keywords based on the closest categories.")

# Upload candidate labels file
uploaded_labels_file = st.file_uploader("Upload Candidate Labels CSV", type=["csv"])
uploaded_keywords_file = st.file_uploader("Upload Keywords File (keywords.txt)", type=["txt"])

if uploaded_labels_file and uploaded_keywords_file:
    # Read candidate labels
    labels_df = pd.read_csv(uploaded_labels_file)
    candidate_labels = labels_df.iloc[:, 0].tolist()  # Assume labels are in the first column
    
    # Encode candidate labels
    st.write("Encoding candidate labels...")
    label_embeddings = model.encode(candidate_labels, convert_to_tensor=True)

    # Read keywords
    st.write("Processing keywords...")
    keywords = uploaded_keywords_file.read().decode('utf-8').strip().split('\n')

    # Prepare results
    results = []

    # Process each keyword
    for keyword in keywords:
        sequence_embedding = model.encode(keyword, convert_to_tensor=True)
        cos_scores = util.cos_sim(sequence_embedding, label_embeddings)[0]

        # Get top 2 results
        top_results = sorted(
            zip(cos_scores, candidate_labels),
            key=lambda x: x[0],
            reverse=True
        )[:2]

        for score, label in top_results:
            results.append({
                'Keyword': keyword,
                'Category': label,
                'Similarity Score': round(score.item(), 3)
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    st.write("Classification Results:")
    st.dataframe(results_df)

    # Download results as CSV
    csv_file = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv_file,
        file_name="classification_results.csv",
        mime="text/csv"
    )
