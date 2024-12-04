import streamlit as st
import pandas as pd
import openai
import tempfile

# Use st.secrets to set your OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=openai.api_key)

# Streamlit app setup
st.title("Keyword Categorizer")
st.write("Upload your keywords file and input your categories.")

# Upload keywords file
uploaded_file = st.file_uploader("Choose a keywords file (TXT)", type="txt")

# Input categories
categories_text = st.text_area("Paste your candidate categories here", height=300)

if uploaded_file and categories_text:
    # Convert categories to list
    candidate_labels = [label.strip() for label in categories_text.splitlines() if label.strip()]
    
    # Read keywords from the uploaded file
    keywords = uploaded_file.getvalue().decode("utf-8").splitlines()

    # Prepare the output data
    results = []

    # Function to use OpenAI for categorization
    def categorize_with_openai(keyword, candidate_labels):
        prompt = f"Classify the keyword '{keyword}' into one or two categories:\n\nCategories:\n- " + "\n- ".join(candidate_labels) + "\n\n"
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            temperature=0.2  # Low temperature for deterministic results
        )
        choices = response.choices[0].message.content.strip().split('\n')[:2]
        return [choice.strip() for choice in choices]

    # Process button
    if st.button("Classify Keywords"):
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
