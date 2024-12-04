import openai
import pandas as pd
import streamlit as st

# Streamlit Secrets Management for OpenAI API Key
api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = api_key

# Streamlit interface
st.title("Keyword Categorization using OpenAI")

# File uploader for candidate labels
st.subheader("Upload Candidate Labels")
label_file = st.file_uploader("Upload your candidate labels file (TXT format)", type="txt")

# File uploader for keywords
st.subheader("Upload Keywords")
keyword_file = st.file_uploader("Upload your keywords file (TXT format)", type="txt")

# Check if both files are uploaded
if label_file is not None and keyword_file is not None:
    # Read the candidate labels from the uploaded file
    candidate_labels = label_file.read().decode("utf-8").splitlines()

    # Read the keywords from the uploaded file
    keywords = keyword_file.read().decode("utf-8").splitlines()

    # Prepare the output data
    results = []

    # Function to use the OpenAI model for categorization
    def categorize_with_openai(keyword, candidate_labels):
        prompt = (f"Given the following categories, classify the following keyword into one appropriate category "
                  f"based on its meaning:\n\nKeyword: {keyword}\nCategories:\n- " +
                  "\n- ".join(candidate_labels) +
                  "\n\n Provide only the category, no other text.")
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Ensure this is the correct model ID
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05  # Low temperature for deterministic results
        )

        choices = response.choices[0].message.content.strip().split('\n')[:2]
        return [choice.strip() for choice in choices]

    # Initialize a progress bar
    progress_bar = st.progress(0)

    # Loop through each keyword
    for index, keyword in enumerate(keywords):
        # Get the top categories using OpenAI
        top_categories = categorize_with_openai(keyword, candidate_labels)

        # Append results to the output list
        for category in top_categories:
            split_category = category.strip('>').strip().split('>')
            results.append({
                'Keyword': keyword,
                'Topic': split_category[0] if len(split_category) > 0 else '',
                'Category': split_category[1] if len(split_category) > 1 else '',
                'Subcategory': split_category[2] if len(split_category) > 2 else '',
                'Subcategory2': split_category[3] if len(split_category) > 3 else ''
            })
        
        # Update progress bar
        progress_bar.progress((index + 1) / len(keywords))

    # Display results as a table and offer download option
    if results:
        df = pd.DataFrame(results)
        st.write("Categorization Results", df)

        # Button to download the dataframe as an Excel file
        @st.cache
        def convert_df_to_excel(df):
            return df.to_excel(index=False)

        if st.download_button(
            label="Download Results as Excel",
            data=convert_df_to_excel(df),
            file_name='classification_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        ):
            st.write("Download initiated!")
    else:
        st.write("No results to display yet.")
