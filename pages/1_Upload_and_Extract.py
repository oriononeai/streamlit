import streamlit as st
import pymupdf
from openai import OpenAI
import os
import json
import pandas as pd
from datetime import datetime

# Initialize session state variables
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = pd.DataFrame(columns=[
        'level', 'paper', 'question_number', 'question', 
        'answer', 'marks', 'question_type', 'astructure'
    ])

if 'review_data' not in st.session_state:
    st.session_state.review_data = pd.DataFrame(columns=[
        'level', 'paper', 'question_number', 'question', 'answer', 
        'marks', 'question_type', 'structured_answer',
        'evaluation_score', 'evaluation_status', 'evaluation_issues',
        'evaluation_suggested_fix', 'review_status',
        'revision_count', 'last_revision', 'revision_error'
    ])

# Define system prompt for initial question extraction
system_prompt = """You are a Science exam paper processing assistant specializing in model answer generation.

You will receive text extracted from a Primary School Science exam paper.
The text includes markers like '--- PAGE X ---' to indicate the start of each page.

Your primary task is to:
1. Extract ALL questions from the paper, regardless of their format or structure
2. Generate appropriate model answers for each question
3. Provide a structured breakdown of each model answer

IMPORTANT: Your response must be a valid JSON array containing objects. Do not use markdown formatting or any other text outside the JSON array.

For every question in the paper, return an object in the JSON array with these fields:
{
    "page_number": integer,  // The page number where the question starts, based on the most recent '--- PAGE X ---' marker
    "question_number": string,  // The question number as it appears in the paper, including any subparts like (a), (b), etc.
    "full_question_text": string,  // The complete question text for that part only
    "question_type": string,  // One of:
        // "explanation" — if the question asks for reasons, causes, scientific processes, effects
        // "experimental_purpose" — if the question asks about the aim of an experiment, controlled variables, fair test conditions
        // "fact_recall" — if the question requires stating a simple fact, definition, or direct knowledge
        // "range" — if the answer to the question is between a range of numbers
        // "multiple_choice" — if the question provides options numbered (1), (2), (3), (4)
    "marks_allocated": integer or null,  // Extract from the number inside square brackets [ ]. Example: [2] becomes 2. If no marks shown, set as null
    "model_answer": string,  // The generated answer content:
        // For multiple_choice questions: Use the option number from the answer key or predict based on the question
        // For other question types: Provide a concise, scientifically accurate answer
    "structured_answer": {  // A dictionary breakdown of the model answer:
        "decision": [],  // List of decisions if applicable
        "cause": [],  // List of scientific causes or triggers
        "effect": [],  // List of outcomes or consequences
        "object": [],  // List of relevant scientific entities involved
        "goal": "",  // Used only in experimental_purpose
        "independent_variable": "",  // Used only in experimental_purpose
        "dependent_variable": "",  // Used only in experimental_purpose
        "controlled_variable": "",  // Used only in experimental_purpose
        "answer": [],  // Used only for fact_recall or multiple_choice if textual
        "number_required": integer  // Only for questions that ask for more than one response
    }
}

Example response format:
[
    {
        "page_number": 1,
        "question_number": "1a",
        "full_question_text": "What is the function of the heart? [2]",
        "question_type": "explanation",
        "marks_allocated": 2,
        "model_answer": "The heart pumps blood to all parts of the body to transport oxygen and nutrients.",
        "structured_answer": {
            "object": ["heart", "blood", "oxygen", "nutrients"],
            "cause": ["heart pumping"],
            "effect": ["blood transported to body parts"],
            "answer": [],
            "number_required": 1
        }
    }
]

IMPORTANT RULES:
- Process EVERY question you find in the paper
- Do not skip any questions or make assumptions about which questions to process
- Look for marks in square brackets [X] at the end of each question
- For questions with subparts (a), (b), etc., process each subpart as a separate question. Name the question as the number with subpart, for example 1a, 1b, etc.
- Do not make assumptions about question numbering or paper structure
- Process all questions regardless of their format or location in the paper
- Return ONLY the JSON array, no other text or formatting"""

def safe_extract_json(text):
    """Extract JSON array from AI response by finding content between first '[' and last ']'."""
    try:
        # Find first '[' and last ']'
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON array found in response")
        
        # Extract just the content between these markers
        json_text = text[start_idx:end_idx+1]
        
        # Clean up any markdown code block markers
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        # Parse the JSON
        return json.loads(json_text)
            
    except Exception as e:
        st.error(f"Error processing AI response: {str(e)}")
        raise ValueError(f"Failed to extract valid JSON: {str(e)}")

# Main Streamlit interface
st.title("Upload PDF and Extract Questions")

# --- User Inputs for Paper Details ---
st.subheader("Select Paper Details")

# Define options
level_options = ["--Select--", "P3", "P4", "P5", "P6"]
school_options = ["--Select--", "ACSP", "ATPS", "HPPS", "MBPS", "MGSP", "NHPS", "ACSJ", "NYPS", "RGPS", "SCGS", "SHPS", "SJIJ", "TNPS", "RSPS", "CHSP", "PCPS", "RSSP"]
year_options = ["--Select--"] + [str(y) for y in range(2027,2019, -1)]

# Create select boxes
level = st.selectbox("Level", options=level_options, key="level_select")
school = st.selectbox("School", options=school_options, key="school_select")
exam_type = st.text_input("Type of Paper, WA2, SA1 etc")
year = st.selectbox("Year", options=year_options, key="year_select")

st.subheader("Upload PDF File")
uploaded_file = st.file_uploader("Upload your Science Paper (.pdf)", type=["pdf"], key="pdf_uploader")
mcq_answer_page = st.number_input("MCQ Answer Key Page Number (Optional)", min_value=1, step=1, value=None, key="mcq_page_input")
mcq_marks_override = st.number_input("Set Fixed Marks for All MCQ Questions (Optional)", min_value=0, step=1, value=None, key="mcq_marks_input")

# Check if all required inputs have valid selections
all_inputs_valid = (
    uploaded_file is not None and
    level != "--Select--" and
    school != "--Select--" and
    exam_type != "" and
    year != "--Select--"
)

paper_code = f"{level}{year}{school}{exam_type}"
st.write(f"Paper Code: {paper_code}")

if all_inputs_valid:
    if st.button("Extract Questions and Generate Answers"):
        with st.spinner("Processing PDF..."):
            try:
                # Process PDF and get structured data
                doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
                full_text = ""
                for i, page in enumerate(doc):
                    full_text += f"\n--- PAGE {i + 1} ---\n"
                    full_text += page.get_text()

                # Get MCQ answers if page specified
                mcq_answer_key_text = None
                if mcq_answer_page is not None:
                    page_idx = int(mcq_answer_page) - 1
                    if 0 <= page_idx < doc.page_count:
                        mcq_answer_key_text = doc[page_idx].get_text()
                
                # Process with AI
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                
                # Debug: Print the prompt being sent to AI
                st.write("Sending prompt to AI...")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_text + (f"\n\n--- MCQ Answer Key ---\n{mcq_answer_key_text}" if mcq_answer_key_text else "")}
                ]
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0
                )
                
                # Debug: Print the raw response from AI
                st.write("Raw AI Response:")
                st.code(response.choices[0].message.content)
                
                # Extract and process structured data
                structured_data = safe_extract_json(response.choices[0].message.content)
                
                # Debug: Print the extracted data
                st.write("Successfully extracted structured data:")
                st.write(f"Number of questions extracted: {len(structured_data)}")
                
                # Create review dataframe with initial state
                review_data = []
                for item in structured_data:
                    q_type = item.get("question_type")
                    ai_marks = item.get("marks_allocated")
                    structured_ans = item.get("structured_answer", {})
                    
                    # Determine marks
                    final_marks = mcq_marks_override if q_type == "multiple_choice" and mcq_marks_override is not None else ai_marks
                    
                    review_data.append({
                        'level': level,
                        'paper': paper_code,
                        'question_number': item.get("question_number"),
                        'question': item.get("full_question_text"),
                        'answer': item.get("model_answer"),
                        'marks': final_marks,
                        'question_type': q_type,
                        'structured_answer': json.dumps(structured_ans),
                        'evaluation_score': None,
                        'evaluation_status': None,
                        'evaluation_issues': None,
                        'evaluation_suggested_fix': None,
                        'review_status': 'pending',
                        'revision_count': 0,
                        'last_revision': None,
                        'revision_error': None
                    })
                
                # Create and update review dataframe
                df = pd.DataFrame(review_data)
                st.session_state.review_data = df
                
                # Create extracted_data dataframe (mirror of pri_sci_paper table)
                extracted_data = df[['level', 'paper', 'question_number', 'question', 'answer', 'marks', 'question_type', 'page_number']].copy()
                extracted_data['astructure'] = df['structured_answer']
                st.session_state.extracted_data = extracted_data
                
                st.success("Extraction completed! You can now review and evaluate the answers.")
                
                # Show navigation button
                if st.button("Go to Review and Re-answer Page"):
                    st.switch_page("pages/2_Review_and_Reanswer.py")
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.session_state.review_data = pd.DataFrame()
                st.session_state.extracted_data = pd.DataFrame()
    
elif not all_inputs_valid and st.button("Extract Questions and Generate Answers"):
    st.warning("Please select Level, School, Exam Type, Year, and upload a PDF file first.")

