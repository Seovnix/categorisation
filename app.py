import streamlit as st
import pandas as pd
import openai
import tempfile

# Set your OpenAI API key from environment variable or directly
openai.api_key = st.secrets("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Streamlit app
st.title("Keyword Categorizer")
st.write("Upload your keywords file and input your categories.")

# Upload file
uploaded_file = st.file_uploader("Choose a keywords file (TXT)", type="txt")

# Input categories
categories_text = st.text_area("Paste your candidate categories here", height=300)
candidate_labels = categories_text.splitlines()

# Process button
process = st.button("Classify Keywords")

if process and uploaded_file and candidate_labels:
    # Read the keywords from the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        keywords = open(tmp_file.name).read().splitlines()

    # Prepare the output data
    results = []

    # Function to use OpenAI for categorization
    def categorize_with_openai(keyword, candidate_labels):
        prompt = f"Given the following categories, classify the keyword '{keyword}' into one or two categories:\n\nCategories:\n- " + "\n- ".join(candidate_labels) + "\n\n"
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2  # Low temperature for deterministic results
        )
        choices = response.choices[0].message.content.strip().split('\n')[:2]
        return [choice.strip() for choice in choices]

    # Loop through each keyword
    with st.spinner('Classifying keywords...'):
        for keyword in keywords:
            top_categories = categorize_with_openai(keyword, candidate_labels)
            for category in top_categories:
                if category:
                    parts = category.split(">")
                    while len(parts) < 5:  # Ensure we have 5 columns
                        parts.append("")
                    results.append({
                        'Keyword': keyword,
                        'Topic': parts[0],
                        'Category': parts[1] if len(parts) > 1 else "",
                        'Subcategory': parts[2] if len(parts) > 2 else "",
                        'Subcategory2': parts[3] if len(parts) > 3 else ""
                    })

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Display results and offer download
    st.success("Classification complete!")
    st.dataframe(df)
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "classification_results.csv", "text/csv")
