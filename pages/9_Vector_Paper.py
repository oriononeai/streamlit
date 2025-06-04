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
        .select('paper','question', 'answer', 'question_type', 'question_number')\
        .eq('paper', paper) \
        .execute()

    data = response.data

    # Manual filtering
    excluded_types = {'MCQ', 'multiple_choice', 'image'}
    filtered_data = [row for row in data if row['question_type'] not in excluded_types]
    
    # Show preview of questions to be processed
    if filtered_data:
        st.info(f"Found {len(filtered_data)} non-MCQ questions to process:")
        st.write("**Questions that will be embedded:**")
        for i, row in enumerate(filtered_data[:5]):  # Show first 5 as preview
            st.write(f"- {row['question_number']}: {row['question'][:100]}...")
        if len(filtered_data) > 5:
            st.write(f"... and {len(filtered_data) - 5} more questions")
    else:
        st.warning("No suitable questions found for embedding.")
        st.stop()

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
        
        try:
            response = openai_client.embeddings.create(
                input=[text], 
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding for text: {e}")
            raise

    # Iterate over the data and generate embeddings
    total_rows = len(filtered_data)
    st.info(f"Starting to generate embeddings for {total_rows} questions...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    success_count = 0
    error_count = 0
    
    for i, row in enumerate(filtered_data):
        try:
            # Update progress
            progress = (i + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"Processing question {i + 1}/{total_rows}: {row['question_number']}")
            
            # Only embed the question text (not the answer)
            text = row['question']
            
            # Skip if question is empty
            if not text or text.strip() == "":
                st.warning(f"Skipping empty question: {row['question_number']}")
                error_count += 1
                continue
                
            embedding = generate_embeddings(text)
            
            # Update the database
            result = supabase.table(TARGET_TABLE)\
                .update({'vsearch': embedding})\
                .eq('question_number', row['question_number']) \
                .eq('paper', row['paper']) \
                .execute()
            
            if result.data:
                success_count += 1
            else:
                st.warning(f"No rows updated for question {row['question_number']}")
                error_count += 1
                
        except Exception as e:
            st.error(f"Error processing question {row['question_number']}: {e}")
            error_count += 1
            continue
    
    # Final status
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    if success_count > 0:
        st.success(f"✅ Successfully processed {success_count} questions")
    if error_count > 0:
        st.warning(f"⚠️ {error_count} questions had errors")
    
    st.success("Embedding generation process completed.")
