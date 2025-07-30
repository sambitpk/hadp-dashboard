import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
import os
import numpy as np

# Initialize Streamlit page configuration
st.set_page_config(page_title="Jansahayak RTI Dashboard", layout="wide")

# Load custom CSS for Marathi font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;700&display=swap');
    .marathi {
        font-family: 'Noto Sans Devanagari', sans-serif;
    }
    .stTextInput > div > input {
        font-family: 'Noto Sans Devanagari', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Translations for bilingual support
translations = {
    "en": {
        "title": "Jansahayak Dashboard",
        "srNo": "Sr. No.",
        "taluka": "Taluka",
        "year": "Year",
        "workName": "Work Name",
        "amount": "Amount (in thousands)",
        "agency": "Agency",
        "type": "Type (A/G)",
        "filterTaluka": "Filter by Taluka",
        "filterYear": "Filter by Year",
        "filterType": "Filter by Type",
        "searchPlaceholder": "Search by work name...",
        "searchButton": "Search",
        "all": "All",
        "interestingFact": "Interesting Fact",
        "tableTitle": "Project Details",
        "costByTaluka": "Total Project Cost by Taluka",
        "projectsByYear": "Number of Projects by Year",
        "projectTypeDist": "Project Type Distribution",
        "language": "Language",
        "english": "English",
        "marathi": "Marathi",
        "errorFile": "Error: HADP_WORK_LIST_MASTER.xlsx not found. Please upload the file.",
        "errorColumns": "Error: Required columns not found in the Excel file.",
        "chatbotTitle": "Jansahayak Chatbot",
        "chatbotPrompt": "Ask a question about the projects...",
        "chatbotError": "Chatbot unavailable: Please configure a valid Hugging Face API key with Write access or check model availability."
    },
    "mr": {
        "title": "जनसहायक डॅशबोर्ड",
        "srNo": "अ. क्र.",
        "taluka": "तालुका",
        "year": "वर्ष",
        "workName": "कामाचे नाव",
        "amount": "प्र.मा रक्कम (हजारात)",
        "agency": "यंत्रणा",
        "type": "प्रकार (A/G)",
        "filterTaluka": "तालुक्याने फिल्टर करा",
        "filterYear": "वर्षानुसार फिल्टर करा",
        "filterType": "प्रकारानुसार फिल्टर करा",
        "searchPlaceholder": "कामाच्या नावाने शोधा...",
        "searchButton": "शोधा",
        "all": "सर्व",
        "interestingFact": "रोचक तथ्य",
        "tableTitle": "प्रकल्प तपशील",
        "costByTaluka": "तालुक्यांनुसार एकूण प्रकल्प खर्च",
        "projectsByYear": "वर्षानुसार प्रकल्पांची संख्या",
        "projectTypeDist": "प्रकल्प प्रकार वितरण",
        "language": "भाषा",
        "english": "इंग्रजी",
        "marathi": "मराठी",
        "errorFile": "त्रुटी: HADP_WORK_LIST_MASTER.xlsx फाइल सापडली नाही. कृपया फाइल अपलोड करा.",
        "errorColumns": "त्रुटी: एक्सेल फाइलमध्ये आवश्यक कॉलम्स सापडले नाहीत。",
        "chatbotTitle": "जनसहायक चॅटबॉट",
        "chatbotPrompt": "प्रकल्पांबद्दल प्रश्न विचारा...",
        "chatbotError": "चॅटबॉट उपलब्ध नाही: कृपया Write परवानगीसह वैध Hugging Face API की कॉन्फिगर करा किंवा मॉडेल उपलब्धता तपासा."
    }
}

# Language display names
language_names = {
    "en": translations["en"]["english"],
    "mr": translations["mr"]["marathi"]
}

# Function to abbreviate numbers
def abbreviate_number(num):
    if pd.isna(num) or num is None:
        return "0"
    if num >= 1000000:
        return f"{num / 1000000:.1f}M"
    if num >= 1000:
        return f"{num / 1000:.1f}K"
    return str(int(num))

# Load and process data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("HADP_WORK_LIST_MASTER.xlsx")
        # Define expected column mappings (Marathi to English keys)
        column_mapping = {
            "अ. क्र.": "srNo",
            "तालुका": "taluka",
            "वर्ष": "year",
            "कामाचे नाव": "workName",
            "प्र.मा रक्कम": "amount",
            "यंत्रणा": "agency",
            "प्रकार (A/G)": "type"
        }
        # Check for available columns
        available_columns = df.columns.tolist()
        missing_columns = [col for col in column_mapping.keys() if col not in available_columns]
        if missing_columns:
            st.error(f"{translations['en']['errorColumns']} Missing: {', '.join(missing_columns)}")
            return pd.DataFrame()
        
        # Rename columns to standardized English keys
        df = df.rename(columns=column_mapping)
        df = df.dropna(subset=["srNo", "amount", "year"])
        df["srNo"] = df["srNo"].astype(int)
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df = df.fillna({"taluka": "", "year": "", "workName": "", "agency": "", "type": ""})
        return df
    except FileNotFoundError:
        st.error(translations["en"]["errorFile"])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Chatbot response function using LangChain
def get_chatbot_response(prompt, df, lang):
    try:
        # Try to get Hugging Face API key from environment variable or secrets
        api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("huggingface", {}).get("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            return translations[lang]["chatbotError"]
        
        # Set up Hugging Face Endpoint
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        llm = HuggingFaceEndpoint(
            repo_id="facebook/bart-large",
            task="text2text-generation",
            max_new_tokens=150,
            temperature=0.5,
            huggingfacehub_api_token=api_key
        )

        # Summarize data for context
        data_summary = df[["taluka", "year", "workName", "amount", "type"]].head(10).to_dict('records')
        prompt_template = PromptTemplate(
            input_variables=["question", "data"],
            template="You are a helpful assistant for the Jansahayak RTI Dashboard. Answer queries about the project data in {lang}. Data summary: {data}\nQuestion: {question}"
        )
        formatted_prompt = prompt_template.format(
            question=prompt,
            data=data_summary,
            lang="English" if lang == "en" else "Marathi"
        )
        response = llm.invoke(formatted_prompt)
        return response.strip()
    except Exception as e:
        return f"{translations[lang]['chatbotError']} Details: {str(e)}"

# Main app
def main():
    # Load data
    df = load_data()
    if df.empty:
        return

    # Language selection
    lang = st.sidebar.selectbox(
        translations["en"]["language"],
        options=["en", "mr"],
        format_func=lambda x: language_names[x]
    )
    t = translations[lang]

    # Header
    st.title(t["title"])

    # Filters and search
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        taluka_filter = st.selectbox(
            t["filterTaluka"],
            options=[""] + sorted(df["taluka"].unique()),
            format_func=lambda x: t["all"] if x == "" else x,
            key="taluka_filter"
        )
    with col2:
        year_filter = st.selectbox(
            t["filterYear"],
            options=[""] + sorted(df["year"].unique()),
            format_func=lambda x: t["all"] if x == "" else x,
            key="year_filter"
        )
    with col3:
        type_filter = st.selectbox(
            t["filterType"],
            options=[""] + sorted(df["type"].unique()),
            format_func=lambda x: t["all"] if x == "" else x,
            key="type_filter"
        )

    # Search input and button
    col4, col5 = st.columns([3, 1])
    with col4:
        search_term = st.text_input(t["searchPlaceholder"], key="search_term", help="Enter work name to search")
    with col5:
        search_button = st.button(t["searchButton"])

    # Filter data
    filtered_df = df.copy()
    if taluka_filter:
        filtered_df = filtered_df[filtered_df["taluka"] == taluka_filter]
    if year_filter:
        filtered_df = filtered_df[filtered_df["year"] == year_filter]
    if type_filter:
        filtered_df = filtered_df[filtered_df["type"] == type_filter]
    if search_button and search_term:
        filtered_df = filtered_df[filtered_df["workName"].str.contains(search_term, case=False, na=False)]

    # Interesting fact
    if not filtered_df.empty:
        max_cost_taluka = df.groupby("taluka")["amount"].sum().idxmax()
        max_cost = df.groupby("taluka")["amount"].sum().max()
        most_frequent_type = df["type"].mode()[0]
        type_count = df["type"].value_counts()[most_frequent_type]
        st.markdown(f"""
            <div class="bg-blue-100 p-4 rounded-lg mb-6">
                <h2 class="text-xl font-semibold text-blue-800">{t["interestingFact"]}</h2>
                <p class="text-gray-700 marathi">
                    {t["taluka"]} <b>{max_cost_taluka}</b> {t["amount"]} <b>{abbreviate_number(max_cost)}</b>.
                    {t["type"]} <b>{most_frequent_type}</b> {t["projectsByYear"]} <b>{type_count}</b>.
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Visualizations
    if not filtered_df.empty:
        st.subheader(t["costByTaluka"])
        cost_by_taluka = df.groupby("taluka")["amount"].sum().reset_index()
        fig_bar = px.bar(cost_by_taluka, x="taluka", y="amount", 
                         labels={"amount": t["amount"], "taluka": t["taluka"]},
                         color_discrete_sequence=["#3B82F6"])
        fig_bar.update_layout(xaxis_tickangle=45, font=dict(family="Noto Sans Devanagari"))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader(t["projectsByYear"])
        projects_by_year = df.groupby("year").size().reset_index(name="count")
        fig_line = px.line(projects_by_year, x="year", y="count", 
                           labels={"count": t["projectsByYear"], "year": t["year"]},
                           color_discrete_sequence=["#10B981"])
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader(t["projectTypeDist"])
        type_dist = df["type"].value_counts().reset_index(name="count")
        type_dist.columns = ["type", "count"]
        fig_pie = px.pie(type_dist, names="type", values="count", 
                         color_discrete_sequence=["#3B82F6", "#10B981"])
        fig_pie.update_layout(font=dict(family="Noto Sans Devanagari"))
        st.plotly_chart(fig_pie, use_container_width=True)

    # Data table
    st.subheader(t["tableTitle"])
    display_df = filtered_df.copy()
    display_df["amount"] = display_df["amount"].apply(abbreviate_number)
    display_df.columns = [t[key] for key in ["srNo", "taluka", "year", "workName", "amount", "agency", "type"]]
    st.dataframe(display_df, use_container_width=True)

    # Chatbot
    st.subheader(t["chatbotTitle"])
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": t["chatbotPrompt"]}]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input(t["chatbotPrompt"]):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Get and display assistant response
        response = get_chatbot_response(prompt, df, lang)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
