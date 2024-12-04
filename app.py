# Import necessary libraries
import streamlit as st
import pandas as pd
from io import BytesIO

# Imports for Google Search Console
import searchconsole
from apiclient import discovery
from google_auth_oauthlib.flow import Flow

# Imports for AgGrid
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode

# Import for OpenAI
import openai

# Import other necessary libraries
import os

###############################################################################
# Streamlit Page Configuration
###############################################################################

# Set the initial layout
if "widen" not in st.session_state:
    layout = "centered"
else:
    layout = "wide" if st.session_state.widen else "centered"

st.set_page_config(
    layout=layout, page_title="GSC Connector & Keyword Categorizer", page_icon="🔌📝"
)

###############################################################################
# Constants
###############################################################################

# Row limit for GSC API calls
ROW_CAP = 25000

# Number of top keywords to categorize
TOP_N_KEYWORDS = 50

###############################################################################
# Tabs Setup
###############################################################################

tab_main, tab_about = st.tabs(["Main", "About"])

with tab_main:
    # Sidebar Configuration
    st.sidebar.image("logo.png", width=290)
    st.sidebar.markdown("")
    st.write("")

    # Retrieve GSC credentials from Streamlit secrets
    client_secret = st.secrets["installed"]["client_secret"]
    client_id = st.secrets["installed"]["client_id"]
    redirect_uri = st.secrets["installed"]["redirect_uris"][0]

    st.markdown("")

    # Initialize session state for OAuth tokens
    if "gsc_token_input" not in st.session_state:
        st.session_state["gsc_token_input"] = ""
    if "gsc_token_received" not in st.session_state:
        st.session_state["gsc_token_received"] = False

    # Callback function for OAuth form submission
    def gsc_form_callback():
        st.session_state.gsc_token_received = True
        query_params = st.experimental_get_query_params()
        if "code" in query_params:
            code = query_params["code"][0]
            st.session_state.gsc_token_input = code
        else:
            st.warning("🚨 Authorization code not found. Please try signing in again.")

    # OAuth Form in Sidebar
    with st.sidebar.form(key="gsc_oauth_form"):
        st.markdown("")

        # Using standard Streamlit link instead of streamlit_elements
        st.markdown(
            f"""
            [🔗 Sign-in with Google](https://accounts.google.com/o/oauth2/auth?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope=https://www.googleapis.com/auth/webmasters.readonly&access_type=offline&prompt=consent)
            """

        )

        # **Added Submit Button**
        submit_oauth = st.form_submit_button(label="Access GSC API")

        if submit_oauth:
            gsc_form_callback()

        st.write("")

        # Instructions
        with st.expander("How to access your GSC data?"):
            st.markdown(
                """
            1. Click on the `Sign-in with Google` link.
            2. You will be redirected to the Google OAuth screen.
            3. Choose the Google account you want to use & click `Continue`.
            4. You will be redirected back to this app.
            5. Click on the "Access GSC API" button.
            6. Voilà! 🙌 
            """
            )
            st.write("")

        # Display OAuth Token (for debugging purposes)
        with st.expander("Check your OAuth token"):
            code = st.text_input(
                "",
                key="gsc_token_input",
                label_visibility="collapsed",
            )

        st.write("")

    # Sidebar Footer
    container3 = st.sidebar.container()
    st.sidebar.write("")
    st.sidebar.caption(
        "Made in 🎈 [Streamlit](https://www.streamlit.io/), by [Charly Wargnier](https://www.charlywargnier.com/)."
    )

    # Initialize df as None
    df = None

    # Handle OAuth and Fetch GSC Data
    try:
        if not st.session_state.gsc_token_received:
            with st.form(key="gsc_initial_form"):
                # Initial Form when OAuth not completed
                st.text_input(
                    "Web property to review (please sign in via Google OAuth first)",
                    value="",
                    disabled=True,
                    label_visibility="collapsed",  # Hide label as it's placeholder
                )
                # **Added Submit Button**
                submit_initial = st.form_submit_button(label="Submit")

                if submit_initial:
                    st.warning("🚨 Please sign in via Google OAuth first.")
        else:
            # Fetch Account and Site List
            @st.cache_data
            def get_account_site_list_and_webproperty(token):
                flow = Flow.from_client_config(
                    {
                        "installed": {
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "redirect_uris": [redirect_uri],
                            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                            "token_uri": "https://accounts.google.com/o/oauth2/token",
                        }
                    },
                    scopes=["https://www.googleapis.com/auth/webmasters.readonly"],
                    redirect_uri=redirect_uri,
                )
                flow.fetch_token(code=token)
                credentials_fetched = flow.credentials
                service = discovery.build(
                    serviceName="webmasters",
                    version="v3",
                    credentials=credentials_fetched,
                    cache_discovery=False,
                )

                account = searchconsole.account.Account(service, credentials_fetched)
                site_list = service.sites().list().execute()
                return account, site_list

            account, site_list = get_account_site_list_and_webproperty(
                st.session_state.gsc_token_input
            )

            # Extract Site URLs
            first_value = list(site_list.values())[0]
            site_urls = [dicts.get("siteUrl") for dicts in first_value if dicts.get("siteUrl")]

            if site_urls:
                container3.info("✔️ GSC credentials OK!")

                # Data Fetching Form
                with st.form(key="gsc_data_form"):
                    selected_site = st.selectbox("Select web property", site_urls)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        dimension = st.selectbox(
                            "Primary Dimension",
                            (
                                "query",
                                "page",
                                "date",
                                "device",
                                "searchAppearance",
                                "country",
                            ),
                            help="Choose your primary dimension for the query.",
                        )
                    with col2:
                        nested_dimension = st.selectbox(
                            "Nested Dimension",
                            (
                                "none",
                                "query",
                                "page",
                                "date",
                                "device",
                                "searchAppearance",
                                "country",
                            ),
                            help="Choose a nested dimension for the query.",
                        )
                    with col3:
                        nested_dimension_2 = st.selectbox(
                            "Second Nested Dimension",
                            (
                                "none",
                                "query",
                                "page",
                                "date",
                                "device",
                                "searchAppearance",
                                "country",
                            ),
                            help="Choose a second nested dimension for the query.",
                        )

                    st.write("")

                    col1, col2 = st.columns(2)

                    with col1:
                        search_type = st.selectbox(
                            "Search Type",
                            ("web", "video", "image", "news", "googleNews"),
                            help="""
                        Specify the search type you want to retrieve:
                        - **Web**: All search results.
                        - **Image**: Image search results.
                        - **Video**: Video search results.
                        - **News**: News search results.
                        - **Google News**: Google News search results.
                        """,
                        )

                    with col2:
                        timescale = st.selectbox(
                            "Date Range",
                            (
                                "Last 7 days",
                                "Last 30 days",
                                "Last 3 months",
                                "Last 6 months",
                                "Last 12 months",
                                "Last 16 months",
                            ),
                            index=0,
                            help="Specify the date range for the data.",
                        )

                        # Convert timescale to days
                        timescale_mapping = {
                            "Last 7 days": -7,
                            "Last 30 days": -30,
                            "Last 3 months": -91,
                            "Last 6 months": -182,
                            "Last 12 months": -365,
                            "Last 16 months": -486,
                        }

                        selected_days = timescale_mapping.get(timescale, -30)

                    st.write("")

                    # Advanced Filters
                    with st.expander("✨ Advanced Filters", expanded=False):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            filter_dim1 = st.selectbox(
                                "Filter Dimension #1",
                                (
                                    "query",
                                    "page",
                                    "device",
                                    "searchAppearance",
                                    "country",
                                ),
                                help="Choose the first dimension to filter.",
                            )
                        with col2:
                            filter_type1 = st.selectbox(
                                "Filter Type #1",
                                (
                                    "contains",
                                    "equals",
                                    "notContains",
                                    "notEquals",
                                    "includingRegex",
                                    "excludingRegex",
                                ),
                                help="Choose the filter type for the first dimension.",
                            )
                        with col3:
                            filter_value1 = st.text_input(
                                "Filter Value #1",
                                "",
                                help="Specify the filter value for the first dimension.",
                            )

                        with col1:
                            filter_dim2 = st.selectbox(
                                "Filter Dimension #2",
                                (
                                    "query",
                                    "page",
                                    "device",
                                    "searchAppearance",
                                    "country",
                                    "none",
                                ),
                                key="filter_dim2",
                                help="Choose the second dimension to filter.",
                            )
                        with col2:
                            filter_type2 = st.selectbox(
                                "Filter Type #2",
                                (
                                    "contains",
                                    "equals",
                                    "notContains",
                                    "notEquals",
                                    "includingRegex",
                                    "excludingRegex",
                                ),
                                key="filter_type2",
                                help="Choose the filter type for the second dimension.",
                            )
                        with col3:
                            filter_value2 = st.text_input(
                                "Filter Value #2",
                                "",
                                key="filter_value2",
                                help="Specify the filter value for the second dimension.",
                            )

                        with col1:
                            filter_dim3 = st.selectbox(
                                "Filter Dimension #3",
                                (
                                    "query",
                                    "page",
                                    "device",
                                    "searchAppearance",
                                    "country",
                                    "none",
                                ),
                                key="filter_dim3",
                                help="Choose the third dimension to filter.",
                            )
                        with col2:
                            filter_type3 = st.selectbox(
                                "Filter Type #3",
                                (
                                    "contains",
                                    "equals",
                                    "notContains",
                                    "notEquals",
                                    "includingRegex",
                                    "excludingRegex",
                                ),
                                key="filter_type3",
                                help="Choose the filter type for the third dimension.",
                            )
                        with col3:
                            filter_value3 = st.text_input(
                                "Filter Value #3",
                                "",
                                key="filter_value3",
                                help="Specify the filter value for the third dimension.",
                            )

                        st.write("")

                    # **Added Submit Button**
                    submit_gsc_data = st.form_submit_button(label="Fetch GSC Data")

                    if submit_gsc_data:
                        webproperty = account[selected_site]

                        # Function to Fetch GSC Data
                        def get_search_console_data(webproperty):
                            if webproperty is not None:
                                q = webproperty.query.search_type(search_type).range("today", days=selected_days).dimension(dimension)

                                # Add nested dimensions if selected
                                if nested_dimension != "none":
                                    q = q.dimension(nested_dimension)
                                if nested_dimension_2 != "none":
                                    q = q.dimension(nested_dimension_2)

                                # Apply filters
                                if filter_dim1 != "none" and filter_value1:
                                    q = q.filter(filter_dim1, filter_value1, filter_type1)
                                if filter_dim2 != "none" and filter_value2:
                                    q = q.filter(filter_dim2, filter_value2, filter_type2)
                                if filter_dim3 != "none" and filter_value3:
                                    q = q.filter(filter_dim3, filter_value3, filter_type3)

                                # Limit the number of rows
                                q = q.limit(ROW_CAP)

                                # Execute the query and convert to DataFrame
                                report = q.get().to_dataframe()
                                return report
                            else:
                                st.warning("No webproperty found")
                                st.stop()

                        df = get_search_console_data(webproperty)

                        if df.empty:
                            st.warning(
                                "🚨 There's no data for your selection, please refine your search with different criteria."
                            )
                            st.stop()
                        else:
                            st.success(f"✅ Data fetched successfully! Total rows: {len(df)}")

                            # Allow user to select metric for top keywords (e.g., clicks, impressions)
                            metric = st.selectbox(
                                "Select metric to determine top keywords",
                                options=["clicks", "impressions", "ctr", "position"],
                                help="Choose the metric based on which top keywords will be selected.",
                            )

                            # Extract top 50 keywords based on selected metric
                            if 'query' in df.columns:
                                top_keywords_df = (
                                    df.groupby('query')[metric]
                                    .sum()
                                    .reset_index()
                                    .sort_values(by=metric, ascending=False)
                                    .head(TOP_N_KEYWORDS)
                                )
                                top_keywords = top_keywords_df['query'].tolist()
                                st.write(f"### Top {TOP_N_KEYWORDS} Keywords based on {metric.capitalize()}")
                                st.dataframe(top_keywords_df)
                            else:
                                st.warning("🚨 The fetched data does not contain the 'query' dimension.")
                                st.stop()

        # ================================== #
        # Separate Form for Keyword Categorization
        # ================================== #
        if df is not None and 'query' in df.columns:
            st.markdown("---")
            st.subheader("Keyword Categorization using OpenAI")

            st.write(
                "Categorize the top keywords into relevant categories using OpenAI's GPT-4 model."
            )

            # Upload Candidate Labels in a Separate Form
            with st.form(key="categorization_form"):
                label_file = st.file_uploader(
                    "Upload your candidate labels file (TXT format)",
                    type="txt",
                )
                # **Added Submit Button**
                submit_categorization = st.form_submit_button(label="Start Categorization")

            if submit_categorization:
                if label_file is not None:
                    # Read Candidate Labels
                    candidate_labels = label_file.read().decode("utf-8").splitlines()

                    if not candidate_labels:
                        st.warning("🚨 The candidate labels file is empty.")
                    else:
                        # Initialize OpenAI API Key
                        openai_api_key = st.secrets["OPENAI"]["OPENAI_API_KEY"]
                        openai.api_key = openai_api_key

                        # Function to Categorize a Single Keyword
                        def categorize_with_openai(keyword, candidate_labels):
                            prompt = (
                                f"Given the following categories, classify the following keyword into one appropriate category "
                                f"based on its meaning:\n\nKeyword: {keyword}\nCategories:\n- " +
                                "\n- ".join(candidate_labels) +
                                "\n\nProvide only the category, no other text."
                            )
                            try:
                                response = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "system", "content": "You are a helpful assistant."},
                                        {"role": "user", "content": prompt},
                                    ],
                                    temperature=0
                                )
                                # Extract the response content
                                content = response.choices[0].message.content.strip()
                                return content
                            except Exception as e:
                                st.error(f"Error categorizing keyword '{keyword}': {e}")
                                return None

                        # Initialize Progress Bar
                        progress_bar = st.progress(0)
                        categorized_results = []

                        # Iterate through Top Keywords and Categorize
                        for idx, keyword in enumerate(top_keywords):
                            category = categorize_with_openai(keyword, candidate_labels)
                            if category:
                                categorized_results.append({
                                    'Keyword': keyword,
                                    'Category': category
                                })
                            progress_bar.progress((idx + 1) / len(top_keywords))

                        # Convert Results to DataFrame
                        if categorized_results:
                            categorized_df = pd.DataFrame(categorized_results)
                            st.success("✅ Keyword categorization completed!")

                            # Display Categorization Results
                            st.write("### Categorization Results")
                            st.dataframe(categorized_df)

                            # Download Button for Excel
                            @st.cache_data
                            def convert_df_to_excel(df):
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    df.to_excel(writer, index=False)
                                processed_data = output.getvalue()
                                return processed_data

                            st.download_button(
                                label="Download Results as Excel",
                                data=convert_df_to_excel(categorized_df),
                                file_name='classification_results.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            )
                        else:
                            st.warning("🚨 No categorization results to display.")
                else:
                    st.warning("🚨 Please upload a candidate labels file (TXT format).")
        # ================================== #
        # End of Keyword Categorization
        # ================================== #

        # ================================== #
        # Display Full GSC Data with Download Option
        # ================================== #
        if df is not None and 'query' in df.columns:
            st.markdown("---")
            st.subheader("Full GSC Data")
            st.write("##### Number of results returned by API call:", len(df.index))

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                st.caption("")
                aggrid_checkbox = st.checkbox(
                    "Ag-Grid mode", help="Tick this box to see your data in Ag-Grid!"
                )
                st.caption("")

            with col2:
                st.caption("")
                st.checkbox(
                    "Widen layout",
                    key="widen",
                    help="Tick this box to switch the layout to 'wide' mode",
                )
                st.caption("")

            # Display DataFrame or AgGrid
            if not aggrid_checkbox:
                @st.cache_data
                def convert_df(df):
                    return df.to_csv(index=False).encode("utf-8")

                csv = convert_df(df)

                st.download_button(
                    label="Download GSC Data as CSV",
                    data=csv,
                    file_name="gsc_data.csv",
                    mime="text/csv",
                )

                st.caption("")
                st.dataframe(df, height=500)
            else:
                df_reset = df.reset_index()

                gb = GridOptionsBuilder.from_dataframe(df_reset)
                gb.configure_default_column(
                    enablePivot=True, enableValue=True, enableRowGroup=True
                )
                gb.configure_selection(selection_mode="multiple", use_checkbox=True)
                gb.configure_side_bar()
                gridOptions = gb.build()

                st.info(
                    """
                    💡 Tip! Hold the '⇧ Shift' key when selecting rows to select multiple rows at once!
                    """
                )

                AgGrid(
                    df_reset,
                    gridOptions=gridOptions,
                    enable_enterprise_modules=True,
                    update_mode=GridUpdateMode.MODEL_CHANGED,
                    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                    height=1000,
                    fit_columns_on_grid_load=True,
                    configure_side_bar=True,
                )
    except ValueError as ve:
        st.warning("⚠️ You need to sign in to your Google account first!")

    except IndexError:
        st.info(
            "⛔ It seems you haven’t correctly configured Google Search Console! Click [here](https://support.google.com/webmasters/answer/9008080?hl=en) for more information on how to get started!"
        )

with tab_about:
    st.write("")
    st.write("")
    st.write(
        """

    #### About this App

    * ✔️ Connect to the [Google Search Console API](https://developers.google.com/webmaster-tools) with OAuth.
    * ✔️ Fetch and view your website's search performance data.
    * ✔️ Automatically extract and categorize the top 50 keywords using OpenAI's GPT-4.
    * ✔️ Download your data and categorization results in CSV and Excel formats.

    ✍️ You can read the blog post [here](https://blog.streamlit.io/p/e89fd54e-e6cd-4e00-8a59-39e87536b260/) for more information.

    #### Going Beyond the `25K` Row Limit

    * There's a `25K` row limit per API call on the [Cloud](https://streamlit.io/cloud) version to prevent crashes.
    * You can remove that limit by forking this code and adjusting the `ROW_CAP` variable in the script.

    #### Kudos

    This app relies on Josh Carty's excellent [Search Console Python wrapper](https://github.com/joshcarty/google-searchconsole). Big kudos to him for creating it!

    #### Questions, Comments, or Report a 🐛?

    * If you have any questions or comments, please DM [me](https://twitter.com/DataChaz). Alternatively, you can ask the [Streamlit community](https://discuss.streamlit.io).
    * If you find a bug, please raise an issue in [Github](https://github.com/CharlyWargnier/google-search-console-connector/pulls).

    #### Known Bugs
    * You can filter any dimension in the table even if the dimension hasn't been pre-selected. I'm working on a fix for this.

    """
    )
