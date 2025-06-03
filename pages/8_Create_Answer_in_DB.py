import streamlit as st
# import pymupdf # Removed
from openai import OpenAI
import os
import json
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import time

# --- Environment Selection (Copied from 4_Structure_DB_OEQ.py) ---
ENV_OPTIONS = ["QA", "PROD"]
# Place selectbox in sidebar
selected_env = st.sidebar.selectbox("Select Environment", ENV_OPTIONS, index=0, key="create_answer_env_select")

# --- Constants (Copied and adapted from 4_Structure_DB_OEQ.py) ---
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

# --- Initialize Supabase Client (Copied from 4_Structure_DB_OEQ.py) ---
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

# --- Initialize OpenAI Client (Copied from 4_Structure_DB_OEQ.py) ---
openai_client: OpenAI | None = None
openai_available = False
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        openai_available = True
    except Exception as e:
        st.sidebar.error(f"Failed to initialize OpenAI client: {e}")
else:
    st.sidebar.warning("OPENAI_API_KEY missing. Cannot generate model answers.")

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

# --- Helper Functions (Copied and adapted from 4_Structure_DB_OEQ.py) ---

def fetch_non_mcq_questions(supabase_client: Client, paper_code: str) -> list[dict]:
    """Fetches non-MCQ questions for a specific paper."""
    if not supabase_available or not supabase_client:
        st.error("Supabase connection not available.")
        return []
    try:
        # Select necessary fields, filter by paper and exclude MCQ/multiple_choice
        response = supabase_client.table(TARGET_TABLE) \
                                  .select("*") \
                                  .eq("paper", paper_code) \
                                  .not_.in_("question_type", ["MCQ", "multiple_choice"]) \
                                  .order("question_number", desc=False) \
                                  .execute()
        if response.data:
            # Filter out rows with missing essential data needed for answer generation
            valid_rows = [
                row for row in response.data 
                if row.get('question') # Ensure question exists
            ]
            count_total = len(response.data)
            count_valid = len(valid_rows)
            if count_total > count_valid:
                 st.warning(f"Fetched {count_total} non-MCQ rows, but {count_total - count_valid} rows are missing 'question' text and will be skipped.")
            return valid_rows
        else:
            st.info(f"No non-MCQ questions found for paper '{paper_code}' or an error occurred.")
            return []
    except Exception as e:
        st.error(f"Error fetching questions for paper {paper_code}: {e}")
        return []

def generate_answer_with_openai(question_data: dict) -> dict:
    """Generate model answer for a single question using OpenAI."""
    if not openai_available or not openai_client:
        return {"error": "OpenAI client not available"}
    
    try:
        question_text = question_data.get('question', 'N/A')
        question_number = question_data.get('question_number', 'Unknown')
        
        # Simplified system prompt for answer generation
        system_prompt = """You are a Primary School Science expert. Generate a concise and accurate model answer for the given question.

Return your response as a JSON object with this format:
{
    "model_answer": "Your concise, scientifically accurate answer here",
    "structured_answer": {
        "decision": [],
        "cause": [],
        "effect": [],
        "object": [],
        "goal": "",
        "independent_variable": "",
        "dependent_variable": "",
        "controlled_variable": "",
        "answer": [],
        "number_required": 
    }
}"""

        user_content = f"""Question: {question_text}
Question Number: {question_number}
Question Type: {question_data.get('question_type', 'Unknown')}
Marks: {question_data.get('marks', 'Not specified')}

Please generate a model answer for this Primary School Science question."""

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        return ai_response
        
    except Exception as e:
        st.error(f"Error generating answer for question {question_data.get('question_number', 'Unknown')}: {e}")
        return {"error": str(e)}

def generate_answers_batch_with_openai(questions_list: list) -> dict:
    """Generate model answers for all questions at once using OpenAI."""
    if not openai_available or not openai_client:
        return {"error": "OpenAI client not available"}
    
    try:
        # Same system prompt logic as individual processing
        system_prompt = """You are a Primary School Science expert teacher. Generate concise and accurate model answers for all the given questions.
            IMPORTANT: Consider the context between related questions (e.g., 29a, 29b, 29c) as they may build upon each other.            
            Return your response as a JSON object with this format:
            {
                "questions_answers": {
                    "question_number_1": {
                        "model_answer": "Your concise, scientifically accurate answer here",
                        "structured_answer": {
                            "decision": [],
                            "cause": [],
                            "effect": [],
                            "object": [],
                            "goal": "",
                            "independent_variable": "",
                            "dependent_variable": "",
                            "controlled_variable": "",
                            "answer": [],
                            "number_required": 1
                        }
                    },
                    "question_number_2": {
                        "model_answer": "Your concise, scientifically accurate answer here",
                        "structured_answer": {
                            "decision": [],
                            "cause": [],
                            "effect": [],
                            "object": [],
                            "goal": "",
                            "independent_variable": "",
                            "dependent_variable": "",
                            "controlled_variable": "",
                            "answer": [],
                            "number_required": 1
                        }
                    }
                    // ... continue for all questions
                }
            }"""

        # Build user content with all questions using same format as individual processing
        questions_text = "Here are all the Primary School Science questions from this paper:\n\n"
        
        for i, question_data in enumerate(questions_list, 1):
            question_text = question_data.get('question', 'N/A')
            question_number = question_data.get('question_number', 'Unknown')
            question_type = question_data.get('question_type', 'Unknown')
            marks = question_data.get('marks', 'Not specified')
            
            questions_text += f"""Question {i}:
            Question Number: {question_number}
            Question Type: {question_type}
            Marks: {marks}
            Question: {question_text}
            
            """

        questions_text += "Please generate model answers for all these Primary School Science questions. Pay attention to any sub-questions that may be related (like 29a, 29b, 29c) and ensure your answers are consistent across related questions."

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": questions_text}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        return ai_response
        
    except Exception as e:
        st.error(f"Error generating batch answers: {e}")
        return {"error": str(e)}

def check_specific_paper(supabase_client: Client, paper_code: str) -> dict:
    """Check if a specific paper exists in the database and return details."""
    if not supabase_available or not supabase_client:
        return {"error": "Supabase connection not available"}
    
    try:
        # Query for the specific paper
        response = supabase_client.table(TARGET_TABLE) \
                                  .select("*") \
                                  .eq("paper", paper_code) \
                                  .execute()
        
        result = {
            "paper_code": paper_code,
            "exists": len(response.data) > 0,
            "total_rows": len(response.data) if response.data else 0,
            "data": response.data[:5] if response.data else []  # Show first 5 rows as sample
        }
        
        if response.data:
            # Get question types breakdown
            question_types = {}
            for row in response.data:
                qtype = row.get('question_type', 'Unknown')
                question_types[qtype] = question_types.get(qtype, 0) + 1
            result["question_types"] = question_types
            
            # Check for non-MCQ questions
            non_mcq = [row for row in response.data if row.get('question_type') not in ['MCQ', 'multiple_choice']]
            result["non_mcq_count"] = len(non_mcq)
        
        return result
        
    except Exception as e:
        return {"error": f"Database query failed: {e}"}
        

def main():
    st.title("Generate Model Answers from Database Questions")

    if not supabase_available:
        st.error(f"Supabase connection to {selected_env} is not configured. Please set the required environment variables.")
        return
    if not openai_available:
        st.error("OpenAI client is not configured. Please set OPENAI_API_KEY environment variable.")
        return 

      
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Paper List"):
        # Clear any potential caches - use correct Streamlit method
        try:
            st.cache_data.clear()
        except:
            pass
        try:
            st.cache_resource.clear()
        except:
            pass
        st.rerun()    
  
    # Paper Code Input Section
    st.subheader("Enter Paper Code")
    
    # Text input for paper code
    paper_code_input = st.text_input(
        "Enter the paper code to process:",
        placeholder="e.g., P62024NHPSPL",
        help="Enter the exact paper code as it appears in the database"
    )
    
    # Validate paper code when entered
    paper_validation_result = None
    if paper_code_input and paper_code_input.strip():
        paper_code = paper_code_input.strip()
        
        # Check if paper exists
        paper_validation_result = check_specific_paper(supabase, paper_code)
        
        if "error" in paper_validation_result:
            st.error(f"❌ Database error: {paper_validation_result['error']}")
        elif not paper_validation_result['exists']:
            st.error(f"❌ Paper '{paper_code}' not found in the database.")
            st.info("💡 Make sure you've selected the correct environment (QA/PROD) and entered the exact paper code.")
        else:
            # Paper exists - show details
            st.success(f"✅ Paper '{paper_code}' found in database!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", paper_validation_result['total_rows'])
            with col2:
                st.metric("Non-MCQ Questions", paper_validation_result.get('non_mcq_count', 0))
            with col3:
                mcq_count = paper_validation_result.get('question_types', {}).get('multiple_choice', 0)
                st.metric("MCQ Questions", mcq_count)
            
            # Show question type breakdown
            if paper_validation_result.get('question_types'):
                st.write("**Question Types:**")
                for qtype, count in paper_validation_result['question_types'].items():
                    st.write(f"- {qtype}: {count}")

    st.markdown("---")

    # Process the paper if it's valid
    if paper_validation_result and paper_validation_result.get('exists') and paper_code_input.strip():
        selected_paper = paper_code_input.strip()
        st.write(f"Processing Paper: **{selected_paper}**")
        
        # Initialize session state for storing processed data
        if 'generated_answers_data' not in st.session_state:
             st.session_state.generated_answers_data = None

        # Fetch and display questions for the selected paper
        st.subheader("Non-MCQ Questions for Selected Paper")
        questions_to_process = fetch_non_mcq_questions(supabase, selected_paper)
        
        if questions_to_process:
            # Display the questions in a dataframe
            questions_df = pd.DataFrame(questions_to_process)
            st.info(f"Found {len(questions_to_process)} non-MCQ questions.")
            st.dataframe(questions_df, height=400)
            
            # Show generate answers button
            if st.button(f"🤖 Generate Model Answers for {selected_paper}"):
                total_questions = len(questions_to_process)
                st.info(f"Generating model answers for {total_questions} questions using batch processing...")

                progress_bar = st.progress(0)
                status_text = st.empty()
                generated_data_list = []
                
                # Use batch processing instead of individual question processing
                status_text.text("Calling OpenAI for batch processing...")
                progress_bar.progress(0.1)
                
                # Generate answers for all questions at once
                ai_result = generate_answers_batch_with_openai(questions_to_process)
                progress_bar.progress(0.8)
                
                status_text.text("Processing AI response...")
                
                if "error" not in ai_result:
                    questions_answers = ai_result.get("questions_answers", {})
                    success_count = 0
                    fail_count = 0
                    
                    # Debug: Show what question keys were returned by AI
                    st.info(f"📊 AI returned answers for keys: {list(questions_answers.keys())}")
                    
                    # Process each question result
                    for question_data in questions_to_process:
                        paper = question_data.get("paper")
                        qn = question_data.get("question_number")
                        question_text = question_data.get("question")
                        
                        # Attempt to derive level from paper_code
                        derived_level = None
                        if paper and len(paper) >= 2 and paper.startswith("P") and paper[1].isdigit():
                            potential_level = paper[:2]
                            if potential_level in ["P3", "P4", "P5", "P6"]:
                                derived_level = potential_level

                        current_result = {
                            "level": question_data.get('level', derived_level),
                            "paper": paper,
                            "question_number": qn,
                            "question": question_text,
                            "original_answer": question_data.get('answer', ''),
                            "marks": question_data.get('marks'),
                            "question_type": question_data.get('question_type'),
                            "generated_answer": "",
                            "structured_answer": "",
                            "status": "failed"
                        }

                        # Look for this question's answer in the batch response
                        question_answer = None
                        # Try different possible keys for the question number
                        possible_keys = [str(qn), f"question_{qn}", f"question_number_{qn}", qn]
                        
                        for key in possible_keys:
                            if key in questions_answers:
                                question_answer = questions_answers[key]
                                break
                        
                        if question_answer and isinstance(question_answer, dict):
                            model_answer = question_answer.get("model_answer", "")
                            structured_answer = question_answer.get("structured_answer", {})
                            
                            current_result.update({
                                "generated_answer": model_answer,
                                "structured_answer": json.dumps(structured_answer),
                                "status": "success"
                            })
                            success_count += 1
                        else:
                            current_result["generated_answer"] = f"Error: No answer found in batch response for question {qn}"
                            fail_count += 1
                        
                        generated_data_list.append(current_result)

                    progress_bar.progress(1.0)
                    status_text.text("Batch processing complete.")
                    
                    # Final Summary
                    st.success(f"Finished batch processing {selected_paper}.")
                    st.write(f"- Successfully generated answers: {success_count}")
                    st.write(f"- Failed to generate answers: {fail_count}")
                    
                    if fail_count > 0:
                        st.warning(f"⚠️ {fail_count} questions failed. This may be due to question number formatting in the AI response.")
                        st.info("💡 You can review the results below and manually check any failed questions.")
                    
                else:
                    # Handle batch processing error
                    st.error(f"❌ Batch processing failed: {ai_result['error']}")
                    st.info("🔄 Consider trying again or checking your OpenAI API connection.")
                    
                    # Create error results for all questions
                    for question_data in questions_to_process:
                        paper = question_data.get("paper")
                        qn = question_data.get("question_number")
                        question_text = question_data.get("question")
                        
                        # Attempt to derive level from paper_code
                        derived_level = None
                        if paper and len(paper) >= 2 and paper.startswith("P") and paper[1].isdigit():
                            potential_level = paper[:2]
                            if potential_level in ["P3", "P4", "P5", "P6"]:
                                derived_level = potential_level

                        current_result = {
                            "level": question_data.get('level', derived_level),
                            "paper": paper,
                            "question_number": qn,
                            "question": question_text,
                            "original_answer": question_data.get('answer', ''),
                            "marks": question_data.get('marks'),
                            "question_type": question_data.get('question_type'),
                            "generated_answer": f"Batch Error: {ai_result['error']}",
                            "structured_answer": "",
                            "status": "failed"
                        }
                        generated_data_list.append(current_result)

                # Store results in session state
                if generated_data_list:
                     st.session_state.generated_answers_data = pd.DataFrame(generated_data_list)
                     
                     # Also populate the review_data and extracted_data for compatibility with other pages
                     review_data_list = []
                     extracted_data_list = []
                     
                     for result in generated_data_list:
                         if result["status"] == "success":
                             review_data_list.append({
                                 'level': result['level'],
                                 'paper': result['paper'],
                                 'question_number': result['question_number'],
                                 'question': result['question'],
                                 'answer': result['generated_answer'],
                                 'marks': result['marks'],
                                 'question_type': result['question_type'],
                                 'structured_answer': result['structured_answer'],
                                 'evaluation_score': None,
                                 'evaluation_status': None,
                                 'evaluation_issues': None,
                                 'evaluation_suggested_fix': None,
                                 'review_status': 'pending',
                                 'revision_count': 0,
                                 'last_revision': datetime.now().isoformat(),
                                 'revision_error': None
                             })
                             
                             extracted_data_list.append({
                                 'level': result['level'],
                                 'paper': result['paper'],
                                 'question_number': result['question_number'],
                                 'question': result['question'],
                                 'answer': result['generated_answer'],
                                 'marks': result['marks'],
                                 'question_type': result['question_type'],
                                 'astructure': result['structured_answer']
                             })
                     
                     st.session_state.review_data = pd.DataFrame(review_data_list)
                     st.session_state.extracted_data = pd.DataFrame(extracted_data_list)
                else:
                     st.session_state.generated_answers_data = None
        else:
            st.warning("No non-MCQ questions found for this paper.")

    # --- Display Generated Results ---
    if st.session_state.get('generated_answers_data') is not None:
         st.markdown("---")
         st.subheader("Generated Model Answers")
         st.info("Review the generated answers below.")

         results_df = st.session_state.generated_answers_data
         st.dataframe(results_df, height=600)
         
         # Navigation to review page
         if not st.session_state.review_data.empty:
             if st.button("📝 Go to Review and Re-answer Page"):
                 st.switch_page("pages/2_Review_and_Reanswer.py")

# --- Run the app ---
if __name__ == "__main__":
    main()

