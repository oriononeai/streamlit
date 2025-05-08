import streamlit as st
from PIL import Image
import streamlit as st
import bcrypt
from dotenv import load_dotenv
import os

# Load environment variables
#load_dotenv("./env/dev.env")
load_dotenv()
st.set_page_config(page_title='Orion One Programs')

logo=Image.open('Logo.png')
# Retrieve credentials from environment variables
USERID = os.getenv("USERID")
PASSWORD_HASH = os.getenv("PASSWORD_HASH").encode('utf-8')

# Initialize session state for authentication if it doesn't exist
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def main():
    if not st.session_state.authenticated:
        login()
    else:
        pages()

def login():
    st.subheader("Please Log In")

    username = st.text_input("User ID")
    password = st.text_input("Password", type="password")

    if st.button("Log In"):
        if username == USERID and bcrypt.checkpw(password.encode('utf-8'), PASSWORD_HASH):
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password. Please try again.")

def pages():
    c1, c2 = st.columns([0.9, 3.2])
    with c1:
        st.caption('')
    with c2:
        st.image(logo, width=300)
        st.title('   Orion One:'
                 'An AI-Enhanced Tutor')
    #st.sidebar.title("Navigation")
    #page = st.sidebar.selectbox("Select a page:", ["Introduction", "CrewAI with Rag", "LLC RAG"])

    #if page == "Introduction":
    st.write("# Welcome to the AI Science Paper Processing Suite!")
    st.markdown("""
    This suite of tools helps process Primary School Science exam papers and manage their questions and model answers in a database.
    Use the sidebar navigation to access the different processing stages. The typical workflow involves the following steps:

    **1. Upload and Initial Extraction (`1_Upload_and_Extract`)**: 
    *   Upload a PDF of the science exam paper.
    *   Provide paper details (Level, School, Year, Exam Type).
    *   The system sends the PDF text to an AI to extract individual questions, determine their type (e.g., "multiple_choice", "explanation", "fact_recall"), allocate marks, and generate an initial model answer string for each question.
    *   This extracted data, including the AI-generated answers, is prepared for database insertion.

    **2. Review and Save Extracted Questions (`0_View_Extracted_Questions`)**: 
    *   Review the data extracted by the AI in an editable table.
    *   Make any necessary corrections to the question text, type, marks, or model answer.
    *   Save the finalized data to the `pri_sci_paper` table in the Supabase database.

    **3. Upload Model Answers via PDF Form (`3_Upload OEQ answers`)**: 
    *   *(Alternative/Supplement to AI generation)* Upload a PDF where model answers have been pre-filled into specific form fields (named like `PaperCode_QuestionNumber_Marks`).
    *   The system extracts the model answer text from each form field.
    *   It fetches the corresponding question text from the Supabase database using the paper code and question number from the field name.
    *   The extracted answers and marks are displayed in an editable table.
    *   You can edit the marks and the model answer text.
    *   Clicking "Update Supabase" saves the potentially edited marks and model answer text back to the corresponding question row in the database.

    **4. Structure OEQ Answers (`4_Structure_DB_OEQ`)**: 
    *   Select a specific paper code from the database.
    *   The system fetches all existing non-multiple-choice questions for that paper along with their current model answer text.
    *   For each question, it sends the question text, current type, and model answer text to an AI.
    *   The AI analyzes the model answer and returns a structured JSON object (containing fields like `decision`, `cause`, `effect`, `answer` list, etc.) and a potentially revised `question_type`.
    *   These structured JSON results and revised question types are displayed in an editable table.
    *   You can review and edit the JSON structure or the question type.
    *   Clicking "Update Supabase" saves the final JSON structure into the `astructure` column and the final question type into the `question_type` column for the relevant questions in the database.
    """)

    # --- Remove old list ---
    # st.write("Welcome to our Orion One AI Marking!")
    # st.write("Use our menu to explore the following functions :")
    # st.write("1.upload_and_extract - Upload PDF file to extract question, and then classifying into questions type and answering concepts")
    # st.write("2. question_splitter (Splits extracted text into questions)")
    # st.write("3. question_type_classifier.py  # (Detects question types)Hybrid Retrieval: Combining dense and sparse strengths")
    # st.write("4. answer_structurer.py   # (Structures model answer JSON)")

    #elif page == "Page 2":
        #st.write("# Welcome to Page 2")

if __name__ == "__main__":
    main()

