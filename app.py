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
st.write("Paste your candidate labels and upload a `keywords.txt` file. The app will classify the keywords based on the closest categories.")

# User input: Candidate labels
labels_input = st.text_area(
    "Paste your candidate labels (one label per line):",
    placeholder="Enter labels here...\nTopic>Category>Subcategory>Subcategory2\nTopic>Category>Subcategory\nTopic>Category",
    height=200,
)

# User input: Keywords file
uploaded_keywords_file = st.file_uploader("Upload Keywords File (keywords.txt)", type=["txt"])

if labels_input and uploaded_keywords_file:
    # Process candidate labels
    candidate_labels = [label.strip() for label in labels_input.split('\n') if label.strip()]
    
    if not candidate_labels:
        st.error("No valid labels provided. Please ensure each label is on a new line.")
    else:
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
                # Split the label into parts
                label_parts = label.split('>')
                topic = label_parts[0] if len(label_parts) > 0 else ""
                category = label_parts[1] if len(label_parts) > 1 else ""
                subcategory = label_parts[2] if len(label_parts) > 2 else ""
                subcategory2 = label_parts[3] if len(label_parts) > 3 else ""

                results.append({
                    'Keyword': keyword,
                    'Topic': topic,
                    'Category': category,
                    'Subcategory': subcategory,
                    'Subcategory2': subcategory2,
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
