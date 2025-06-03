import streamlit as st
import openai
from supabase import create_client, Client
import os
import truststore
truststore.inject_into_ssl()

# --- Environment Selection ---
ENV_OPTIONS = ["QA", "PROD"]
# Place selectbox in sidebar
selected_env = st.sidebar.selectbox("Select Environment", ENV_OPTIONS, index=0, key="structure_env_select")

# --- Constants ---
# Conditional Supabase credentials based on selected_env
if selected_env == "QA":
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
elif selected_env == "PROD":
    SUPABASE_URL = os.getenv("SUPABASEO1_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASEO1_API_KEY")
else: # Default to QA
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_API_KEY = os.getenv("SUPABASE_API_KEY")
    st.sidebar.warning(f"Unknown environment '{selected_env}'. Defaulting to QA.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_TABLE = "pri_sci_paper"

# --- Initialize Supabase Client ---
supabase: Client | None = None
supabase_available = False
try:
    if SUPABASE_URL and SUPABASE_API_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)
        supabase_available = True
        st.sidebar.success(f"Supabase connection for {selected_env} established.")
    else:
        st.sidebar.warning(f"Supabase URL/Key missing for {selected_env}. Cannot connect to DB.")
except Exception as e:
    st.sidebar.error(f"Supabase connection failed: {e}")

# Streamlit app
st.title("Generate and Save OpenAI Embeddings")

# Input parameter
paper = st.text_input("Enter the paper parameter:")

if st.button("Generate Embeddings") and paper:
    # Fetch data from the table
    response = supabase.table('pri_sci_paper')\
        .select('question', 'answer', 'question_type', 'question_number')\
        .eq('paper', paper) \
        .not_('question_type', 'in','("MCQ","multiple_choice","image")') \
        .execute()

    data = response.data

    # Function to generate embeddings

    # --- Initialize OpenAI Client ---
    openai_client: openai.OpenAI | None = None
    openai_available = False
    if OPENAI_API_KEY:
        try:
            openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            openai_available = True
        except Exception as e:
            st.error(f"Failed to initialize OpenAI client: {e}")


    # --- Function to Generate Embeddings ---
    def generate_embeddings(text: str):
        if not openai_available:
            raise RuntimeError("OpenAI client is not available.")

        response = openai_client.embeddings.create(input = [text], model = "text-embedding-ada-002")
        return response.data[0].embedding

    # Iterate over the data and generate embeddings
    for row in data:
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        embedding = generate_embeddings(text)
        supabase.table(TARGET_TABLE)\
            .update({'vsearch': embedding})\
            .eq('question_number', row['question_number']) \
            .eq('paper', row['paper']) \
                    .execute()

    st.success("Embeddings generated and saved successfully.")
