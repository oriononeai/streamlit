import streamlit as st
import json
import pandas as pd
from datetime import datetime
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load marking prompt
try:
    with open("prompt.txt", "r", encoding='utf-8') as f:
        marking_prompt = f.read()
except Exception as e:
    st.error(f"Error reading prompt.txt: {str(e)}")
    st.stop()

# Define system prompt for evaluation
system_prompt = """You are a Science exam paper evaluation assistant. 
IMPORTANT: return an object in the JSON array with these fields. Do NOT use markdown formatting or any extra text outside the JSON array.

Your job is to:
1. Evaluate the model answer using the provided marking instructions
2. If it is not good enough (status = "fail"), return a revised model answer and breakdown that would pass

Return the following structure:
[
    {
        "evaluation": {
            "score": string (e.g., "1 mark", "2 marks"),
            "status": string ("pass" or "fail"),
            "issues": array of strings (reasons why it failed or empty if passed)
        },
        "model_answer": {
            "answer": string (the improved or approved model answer),
            "structured_breakdown": {
                "decision": array of strings,
                "cause": array of strings,
                "effect": array of strings,
                "object": array of strings,
                "goal": string,
                "independent_variable": string,
                "dependent_variable": string,
                "controlled_variable": string
            }
        }
    }
]

Additional Instructions:
- If the answer fails, generate a scientifically correct, complete, and structured improved model answer.
- Your revised model answer must include all the components needed based on the question type and marking instructions.

Marking instructions:
""" + marking_prompt

def safe_extract_json(text):
    """Extract JSON from AI response by finding content between first '{' or '[' and last '}' or ']'."""
    try:
        # Debug: Show raw response and its length
        st.write("Raw AI Response:")
        st.code(text)
        st.write(f"Response length: {len(text)}")
        st.write("First 10 characters:", repr(text[:10]))
        st.write("Last 10 characters:", repr(text[-10:]))
        
        # Find first '{' or '[' and last '}' or ']'
        start_bracket = text.find('{')
        start_array = text.find('[')
        end_bracket = text.rfind('}')
        end_array = text.rfind(']')
        
        # Determine which markers to use
        if start_bracket != -1 and (start_array == -1 or start_bracket < start_array):
            start_idx = start_bracket
            end_idx = end_bracket
            st.write(f"Using object format ({{}})")
        else:
            start_idx = start_array
            end_idx = end_array
            st.write(f"Using array format ([])")
        
        st.write(f"Found start marker at position: {start_idx}")
        st.write(f"Found end marker at position: {end_idx}")
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON found in response")
        
        # Extract just the content between these markers
        json_text = text[start_idx:end_idx+1]
        
        # Debug: Show extracted JSON text
        st.write("Extracted JSON text:")
        st.code(json_text)
        
        # Clean up any markdown code block markers
        json_text = json_text.replace('```json', '').replace('```', '').strip()
        
        # Debug: Show cleaned JSON text
        st.write("Cleaned JSON text:")
        st.code(json_text)
        
        # Parse the JSON
        result = json.loads(json_text)
        
        # If result is a single object, wrap it in a list
        if isinstance(result, dict):
            result = [result]
        elif not isinstance(result, list):
            raise ValueError("Response must be a JSON object or array")
            
        return result
            
    except Exception as e:
        st.error(f"Error processing AI response: {str(e)}")
        raise ValueError(f"Failed to extract valid JSON: {str(e)}")

def evaluate_and_provide_model_answer(row):
    """Evaluate a given answer and provide a model answer if necessary."""
    try:
        # Convert row to dict if it's a Series
        if isinstance(row, pd.Series):
            row = row.to_dict()
        
        # Clean up data
        question = str(row.get('question', ''))
        answer = str(row.get('answer', ''))
        question_type = str(row.get('question_type', ''))
        marks = row.get('marks')
        if pd.isna(marks):
            marks = None
        
        # Construct user prompt
        user_prompt = f"""Question: {question}
Question Type: {question_type}
Marks: {marks}
Current Answer: {answer}

Please evaluate this answer and provide a model answer if it fails evaluation."""

        # Get AI response
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        
        # Parse response using safe_extract_json
        result = safe_extract_json(response.choices[0].message.content)
        
        # Get the first (and should be only) object from the array
        if not isinstance(result, list) or len(result) == 0:
            raise ValueError("Response must contain at least one object")
        result = result[0]
        
        # Validate required fields
        required_fields = ['evaluation', 'model_answer']
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field '{field}' in AI response")
        
        # Ensure evaluation has required subfields
        eval_fields = ['score', 'status', 'issues']
        for field in eval_fields:
            if field not in result['evaluation']:
                raise ValueError(f"Missing required field 'evaluation.{field}' in AI response")
        
        # Ensure model_answer has required subfields
        model_fields = ['answer', 'structured_breakdown']
        for field in model_fields:
            if field not in result['model_answer']:
                raise ValueError(f"Missing required field 'model_answer.{field}' in AI response")
        
        return result
            
    except Exception as e:
        st.error(f"Error evaluating answer: {str(e)}")
        raise

def process_questions_for_evaluation(df):
    """Process questions for evaluation and reanswering"""
    # Create a copy of the DataFrame
    df = df.copy()
    
    # Convert NULL strings to None
    for col in df.columns:
        df[col] = df[col].replace('NULL', None)
    
    # Reset index to ensure we have a clean numeric index
    df = df.reset_index(drop=True)
    
    # Get questions to process (non-MCQ with pending or failed status)
    questions_to_process = df[
        (df['question_type'] != 'multiple_choice') & 
        ((df['evaluation_status'].isna()) | (df['evaluation_status'] == 'fail'))
    ]
    
    if questions_to_process.empty:
        return df, "No questions to process."
    
    # Process each question
    for idx, row in questions_to_process.iterrows():
        try:
            result = evaluate_and_provide_model_answer(row)
            
            # Update the DataFrame with evaluation results
            df.iloc[idx, df.columns.get_loc('evaluation_score')] = result['evaluation']['score']
            df.iloc[idx, df.columns.get_loc('evaluation_status')] = result['evaluation']['status']
            df.iloc[idx, df.columns.get_loc('evaluation_issues')] = json.dumps(result['evaluation']['issues'])
            
            # If the answer failed evaluation, update with the model answer
            if result['evaluation']['status'] == 'fail':
                df.iloc[idx, df.columns.get_loc('answer')] = result['model_answer']['answer']
                df.iloc[idx, df.columns.get_loc('structured_answer')] = json.dumps(result['model_answer']['structured_breakdown'])
                # Store the first issue as the suggested fix
                df.iloc[idx, df.columns.get_loc('evaluation_suggested_fix')] = result['evaluation']['issues'][0] if result['evaluation']['issues'] else ""
            
            df.iloc[idx, df.columns.get_loc('review_status')] = 'completed' if result['evaluation']['status'] == 'pass' else 'manual_review'
            df.iloc[idx, df.columns.get_loc('revision_count')] = df.iloc[idx, df.columns.get_loc('revision_count')] + 1
            df.iloc[idx, df.columns.get_loc('last_revision')] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df.iloc[idx, df.columns.get_loc('revision_error')] = None
            
        except Exception as e:
            df.iloc[idx, df.columns.get_loc('revision_error')] = str(e)
            st.error(f"Error processing question {row.get('question_number', 'unknown')}: {str(e)}")
    
    return df, f"Processed {len(questions_to_process)} questions."

# Main Streamlit interface
st.title("Review and Re-answer Questions")

# Check if we have data in session state
if 'review_data' not in st.session_state or st.session_state.review_data.empty:
    st.warning("No data available for review. Please process a paper first.")
    if st.button("Go to Upload Page"):
        st.switch_page("pages/1_Upload_and_Extract.py")
    st.stop()

# Get the review data
df = st.session_state.review_data

# Display current review data
st.subheader("Current Review Data")
st.dataframe(
    df[['question_number', 'question_type', 'question', 'answer', 'marks', 
        'evaluation_score', 'evaluation_status', 'evaluation_issues', 
        'review_status', 'revision_count']],
    column_config={
        "question": st.column_config.TextColumn("Question", width="large"),
        "answer": st.column_config.TextColumn("Answer", width="large"),
        "evaluation_issues": st.column_config.TextColumn("Issues", width="large"),
        "evaluation_score": st.column_config.TextColumn("Score"),
        "evaluation_status": st.column_config.TextColumn("Status"),
        "review_status": st.column_config.TextColumn("Review Status"),
        "revision_count": st.column_config.NumberColumn("Revisions")
    },
    hide_index=True
)

# Evaluation button
if st.button("Evaluate and Re-answer Questions"):
    with st.spinner("Evaluating and re-answering questions..."):
        updated_df, message = process_questions_for_evaluation(df)
        st.session_state.review_data = updated_df
        st.success(message)
        st.rerun()

# Copy to extracted data button
if st.button("Copy to Extracted Data"):
    with st.spinner("Copying data..."):
        # Initialize extracted_data if it doesn't exist
        if 'extracted_data' not in st.session_state:
            st.session_state.extracted_data = pd.DataFrame()
        
        # Get the columns we want to copy
        columns_to_copy = [
            'level', 'paper', 'question_number', 'question', 
            'answer', 'marks', 'question_type'
        ]
        
        # Create a copy of the selected columns
        extracted_df = df[columns_to_copy].copy()
        
        # Add astructure field (copy from structured_answer)
        extracted_df['astructure'] = df['structured_answer']
        
        # Update session state
        st.session_state.extracted_data = extracted_df
        st.success(f"Successfully copied {len(extracted_df)} questions to extracted data.")

# Navigation button
if st.button("Back to Upload Page"):
    st.switch_page("pages/1_Upload_and_Extract.py") 