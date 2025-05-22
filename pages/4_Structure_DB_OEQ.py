import streamlit as st
import os
import pandas as pd
import openai
import json
import re # For safe_extract_json
from supabase import create_client, Client
import time # For potential delays

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

# --- Initialize OpenAI Client ---
openai_client: openai.OpenAI | None = None
openai_available = False
if OPENAI_API_KEY:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        openai_available = True
    except Exception as e:
        st.sidebar.error(f"Failed to initialize OpenAI client: {e}")
else:
    st.sidebar.warning("OPENAI_API_KEY missing. Cannot structure answers.")

# --- Helper Functions ---

def safe_extract_json(text):
    """
    Extract JSON object from first `{` to last `}` for potentially noisy AI response.
    Adjusted to expect a single JSON object, not necessarily an array.
    """
    if not isinstance(text, str):
        return None
    try:
        # Find the first opening curly brace
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        # Find the last closing curly brace
        end_idx = text.rfind('}')
        if end_idx == -1 or end_idx < start_idx:
            return None
            
        # Extract the potential JSON string
        json_text = text[start_idx : end_idx + 1]
        
        # Attempt to load the extracted string as JSON
        return json.loads(json_text)
    except (ValueError, json.JSONDecodeError) as e:
        st.warning(f"Could not parse extracted JSON: {e}\nContent slice: {json_text[:500]}...") # Show preview
        return None


def fetch_distinct_papers(supabase_client: Client) -> list[str]:
    """Fetches distinct paper codes from the Supabase table."""""
    papers = ["--Select Paper--"]
    if not supabase_available or not supabase_client:
        st.error("Supabase connection not available.")
        return papers
    try:
        response = supabase_client.table(TARGET_TABLE).select("paper", count='exact').execute()
        # Handle potential differences in response structure for distinct
        # A common way is to fetch all and get unique values
        response_all = supabase_client.table(TARGET_TABLE).select("paper").execute()
        if response_all.data:
             distinct_papers = sorted(list(set(item['paper'] for item in response_all.data if item.get('paper'))))
             papers.extend(distinct_papers)
        else:
             st.warning(f"Could not fetch distinct papers or table is empty. Response: {response_all}")

    except Exception as e:
        st.error(f"Error fetching distinct papers: {e}")
    return papers

def fetch_oeq_questions(supabase_client: Client, paper_code: str) -> list[dict]:
    """Fetches non-MCQ questions for a specific paper."""""
    if not supabase_available or not supabase_client:
        st.error("Supabase connection not available.")
        return []
    try:
        # Select necessary fields, filter by paper and exclude MCQ
        st.write(f"paper code is {paper_code}")
        response = supabase_client.table(TARGET_TABLE) \
                                  .select("paper, question_number, question, question_type, answer") \
                                  .eq("paper", paper_code) \
                                  .neq("question_type", "multiple_choice") \
                                  .neq("question_type", "MCQ") \
                                  .execute()
        if response.data:
            # Filter out rows with missing essential data needed for structuring
            valid_rows = [
                row for row in response.data 
                if row.get('question') and row.get('answer') # Ensure question and answer exist
            ]
            count_total = len(response.data)
            count_valid = len(valid_rows)
            if count_total > count_valid:
                 st.warning(f"Fetched {count_total} non-MCQ rows, but {count_total - count_valid} rows are missing 'question' or 'answer' text and will be skipped.")
            return valid_rows
        else:
            st.info(f"No non-multiple choice questions found for paper '{paper_code}' or an error occurred.")
            # st.write(response) # Optional: for debugging
            return []
    except Exception as e:
        st.error(f"Error fetching questions for paper {paper_code}: {e}")
        return []

# --- Adapted System Prompt (Focus on structuring based on provided info) ---
AI_SYSTEM_PROMPT = """You are a science exam marking assistant.

You will be provided with data for **a single question** retrieved from a database containing:
- The Full Question Text
- The current Question Type classification
- The raw Model Answer text.

Your task is to convert this data into a clean structured JSON object containing the potentially revised question type and the structured model answer, matching the expected science marking answer schema.

---

Output a **single JSON object** with ONLY the following two top-level fields:

| Field | Meaning |
|:--|:--|
| `question_type` | Verify the provided type or re-classify into one: "explanation", "fact_recall", "experimental_purpose". Use the original type if unsure or if it seems correct. |
| `model_answer` | A structured object as explained below. |

Inside the `model_answer` object, include all applicable fields below:

| Field | Rule |
|:--|:--|
| `decision` | List of possible decisions. If multiple options in the provided Model Answer (separated by `|`), split into a list. Leave empty list `[]` if not applicable. |
| `cause` | List of causes. If multiple options separated by `|`, split into a list. |
| `effect` | List of effects. |
| `object` | List of affected objects. |
| `goal` | A string representing the aim (for experimental questions). Leave blank `""` if not applicable. |
| `independent_variable` | A string representing what is changed in the experiment. |
| `controlled_variable` | A string representing what is kept constant in the experiment. |
| `dependent_variable` | A string representing what is measured or observed. |
| `answer` | List of direct answers (for fact recall). If multiple options separated by `|`, split into a list. |
| `number_required` | Integer. Set only if the provided Question Text clearly asks for naming two, three, etc. If not mentioned, omit this field. |

---

**Special Handling Instructions:**

- If the provided Model Answer includes multiple correct options, split by the `|` symbol.
- Clean and trim each split part carefully (remove spaces before/after text).
- If a sub-field within `model_answer` is not applicable for the question type, leave it as:
  - `[]` for lists (decision, cause, effect, object, answer),
  - `""` for strings (goal, independent_variable, controlled_variable, dependent_variable).
- `number_required` must be manually interpreted based on the provided full question text if it mentions "two", "three", etc.
# - Always set `marks_allocated` to `null`. (No longer needed as it's not an output field)

---

**Formatting Rules:**

- Always output **only the single, raw JSON object** containing only the `question_type` and `model_answer` fields.
- Do not output extra commentary, instructions, or markdown formatting like ```json.
- Only return the final JSON structure, no headings, no extra text.

---

**Example Input Context (User Message):**
# Question Number: 3a (No longer sent)
Full Question Text: Name two other physical factors that can affect the temperature of air in the two habitats.
Question Type: fact_recall
Model Answer: Humidity | Wind speed | Cloud cover | Amount of sunlight | Rainfall

**Example Output (Your Response):**
{
    "model_answer": {
    "decision": [],
    "cause": [],
    "effect": [],
    "object": [],
    "goal": "",
    "independent_variable": "",
    "controlled_variable": "",
    "dependent_variable": "",
    "answer": ["Humidity", "Wind speed", "Cloud cover", "Amount of sunlight", "Rainfall"],
    "number_required": 2
  }
}
"""

def structure_answer_with_openai(paper: str, qn: str, question: str, q_type: str, answer: str) -> str | None:
    """Calls OpenAI API synchronously to structure the answer text."""""
    if not openai_available or not openai_client:
        st.error("OpenAI client not available.")
        return None
    if not answer or not isinstance(answer, str) or not answer.strip():
        st.warning(f"Skipping QN {qn} for Paper {paper}: Invalid or empty answer text provided.")
        return None # Skip if answer is empty

    user_message = f"""
Structure the model answer based on the following information:

Question Text: {question}
Provided Question Type: {q_type}
Model Answer Text to Structure:
{answer}
"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o", # Or your preferred model
            messages=[
                {"role": "system", "content": AI_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1 # Allow slight flexibility for structuring nuances
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        st.error(f"OpenAI API call failed for QN {qn} (Paper {paper}): {e}")
        return None


def update_supabase_structure(supabase_client: Client, paper: str, qn: str, astructure_json: dict, question_type: str) -> bool:
    """Updates the astructure and question_type for a specific question in Supabase."""""
    if not supabase_available or not supabase_client:
        st.error("Supabase connection not available.")
        return False
        
    try:
        update_payload = {
            "astructure": astructure_json, # Already a dict from safe_extract_json
            "question_type": question_type # Use the type confirmed/refined by AI
        }
        response = supabase_client.table(TARGET_TABLE) \
                                  .update(update_payload) \
                                  .match({'paper': paper, 'question_number': qn}) \
                                  .execute()

        # Check for errors after update
        if hasattr(response, 'error') and response.error:
             st.error(f"Supabase update error for QN {qn} (Paper {paper}): {response.error.message}")
             return False
        # Can check response.data length or status if API guarantees it
        # For now, assume success if no error
        return True 
    except Exception as e:
        st.error(f"Exception during Supabase update for QN {qn} (Paper {paper}): {e}")
        return False

# --- Streamlit App Main Logic ---
def main():
    st.title("Structure OEQ Answers in Database")

    if not supabase_available:
        st.error(f"Supabase connection to {selected_env} is not configured. Please set the required environment variables.")
        return
    if not openai_available:
        st.error("OpenAI client is not configured. Please set OPENAI_API_KEY environment variable.")
        # Allow proceeding without OpenAI to see papers, but disable button
        # return 

    papers = fetch_distinct_papers(supabase)
    selected_paper = st.selectbox("Select Paper Code to Process:", options=papers)

    st.markdown("---")

    if selected_paper != "--Select Paper--":
        st.write(f"Selected Paper: **{selected_paper}**")
        
        # Disable button if OpenAI isn't available
        process_button_disabled = not openai_available 
        if process_button_disabled:
             st.warning("OpenAI is not available. Cannot process structures.")

        # Initialize session state for storing processed data
        if 'processed_oeq_data' not in st.session_state:
             st.session_state.processed_oeq_data = None

        if st.button(f"Fetch and Structure OEQ Answers for {selected_paper}", disabled=process_button_disabled):
            questions_to_process = fetch_oeq_questions(supabase, selected_paper)

            if not questions_to_process:
                st.warning("No suitable questions found to process for this paper.")
                return

            total_questions = len(questions_to_process)
            st.info(f"Found {total_questions} OEQ questions to structure.")

            progress_bar = st.progress(0)
            status_text = st.empty()
            success_count = 0
            fail_count = 0
            json_fail_count = 0
            processed_data_list = [] # Store results here
            
            # Use st.status for cleaner logging during processing
            with st.status(f"Processing {total_questions} questions...", expanded=True) as status_container:
                for i, question_data in enumerate(questions_to_process):
                    paper = question_data.get("paper")
                    qn = question_data.get("question_number")
                    question_text = question_data.get("question")
                    original_q_type = question_data.get("question_type", "unknown") # Handle missing type
                    answer_text = question_data.get("answer")

                    if not all([paper, qn, question_text, answer_text]):
                         st.warning(f"Skipping row {i+1}: Missing essential data (paper, qn, question, or answer).")
                         fail_count +=1
                         continue # Skip if core data is missing

                    status_text.text(f"Processing {i+1}/{total_questions}: Paper {paper}, QN {qn}...")
                    st.write(f"Processing QN {qn}: Calling OpenAI...")

                    # 1. Call OpenAI
                    raw_ai_response = structure_answer_with_openai(
                        paper=paper,
                        qn=qn,
                        question=question_text,
                        q_type=original_q_type,
                        answer=answer_text
                    )

                    current_result = {
                        "paper": paper,
                        "question_number": qn,
                        "question": question_text,
                        "original_answer": answer_text, # Keep original answer for reference
                        "question_type": original_q_type, # Start with original type
                        "astructure": None # Placeholder for the structured JSON dict
                    }

                    if raw_ai_response:
                        # 2. Safely extract JSON
                        st.write(f"Processing QN {qn}: Extracting JSON...")
                        structured_json = safe_extract_json(raw_ai_response)

                        if structured_json and isinstance(structured_json, dict):
                             final_q_type = structured_json.get("question_type", original_q_type) # Default to original if AI doesn't provide one
                             current_result["astructure"] = structured_json # Store the dict
                             current_result["question_type"] = final_q_type # Store potentially updated type
                             success_count += 1 # Count success if JSON parsed
                        else:
                            st.warning(f"Failed to extract valid JSON object for QN {qn} (Paper {paper}). Skipping database update.")
                            st.text_area(f"Raw AI Response for QN {qn}", raw_ai_response, height=100)
                            json_fail_count += 1
                            fail_count += 1
                            current_result["astructure"] = {"error": "Failed to parse JSON", "raw_response": raw_ai_response}
                    else:
                        # OpenAI call failed or skipped (error logged within function)
                         st.write(f"Processing QN {qn}: OpenAI call failed or skipped.")
                         fail_count += 1
                         current_result["astructure"] = {"error": "OpenAI call failed or skipped"}
                        
                    processed_data_list.append(current_result)

                    # Update progress bar
                    progress = (i + 1) / total_questions
                    progress_bar.progress(min(progress, 1.0))
                    
                    # Optional delay to help with rate limits if needed
                    # time.sleep(0.5) 

            # Final Summary
            status_text.text("Processing Complete.")
            st.success(f"Finished processing {selected_paper}.")
            st.write(f"- Successfully structured and updated: {success_count}")
            st.write(f"- Failed due to OpenAI/Update errors: {fail_count - json_fail_count}")
            st.write(f"- Failed due to invalid JSON response: {json_fail_count}")
            status_container.update(label="Processing Complete!", state="complete", expanded=False)

            # Store results in session state if successful processing occurred
            if processed_data_list:
                 st.session_state.processed_oeq_data = pd.DataFrame(processed_data_list)
            else:
                 st.session_state.processed_oeq_data = None # Clear if no data

    # --- Display Editor and Update Button outside the processing button block ---
    if st.session_state.get('processed_oeq_data') is not None:
         st.markdown("---")
         st.subheader("Edit Structured Answers")
         st.info("Review the structured answers below. You can edit the 'astructure' JSON and 'question_type'.")

         df_to_edit = st.session_state.processed_oeq_data.copy()

         # Convert dict to JSON string for display/editing
         if 'astructure' in df_to_edit.columns:
             df_to_edit['astructure'] = df_to_edit['astructure'].apply(
                 lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else str(x)
             )

         edited_df = st.data_editor(
             df_to_edit,
             height=600,
             num_rows="fixed",
             column_config={
                 "paper": st.column_config.TextColumn(disabled=True),
                 "question_number": st.column_config.TextColumn("QN", disabled=True),
                 "question": st.column_config.TextColumn(disabled=True),
                 "original_answer": st.column_config.TextColumn("Original Answer", disabled=True),
                 "question_type": st.column_config.SelectboxColumn(
                     "Question Type (Editable)", 
                     options=["explanation", "fact_recall", "experimental_purpose", "unknown"], 
                     required=True
                 ),
                 "astructure": st.column_config.TextColumn("Structured Answer (JSON - Editable)")
             },
             disabled=["paper", "question_number", "question", "original_answer"],
             key="structure_editor"
         )

         st.markdown("---")
         if st.button("Update Supabase with Edited Structures"):
             if edited_df is not None:
                 update_success_count = 0
                 update_fail_count = 0
                 json_parse_errors = 0

                 with st.spinner("Updating Supabase..."):
                     for index, row in edited_df.iterrows():
                         paper = row["paper"]
                         qn = row["question_number"]
                         final_q_type = row["question_type"]
                         astructure_str = row["astructure"]

                         # Try to parse the edited JSON string back to dict
                         try:
                             # Handle potential non-string data (e.g., if initial processing failed)
                             if isinstance(astructure_str, dict):
                                 astructure_dict = astructure_str # Already a dict
                             elif isinstance(astructure_str, str):
                                 astructure_dict = json.loads(astructure_str) 
                             else:
                                  raise ValueError("Unexpected data type in astructure column")
                             
                             # Skip rows where initial structuring failed (contains error key)
                             if isinstance(astructure_dict, dict) and "error" in astructure_dict:
                                  st.warning(f"Skipping QN {qn} (Paper {paper}): Row contains processing errors.")
                                  continue

                             # Call update function
                             update_successful = update_supabase_structure(
                                 supabase, 
                                 paper, 
                                 qn, 
                                 astructure_dict, # Pass the parsed dict
                                 final_q_type
                             )
                             if update_successful:
                                 update_success_count += 1
                             else:
                                 update_fail_count += 1

                         except json.JSONDecodeError as json_e:
                              st.error(f"Invalid JSON format for QN {qn} (Paper {paper}): {json_e}. Skipping update for this row.")
                              json_parse_errors += 1
                              update_fail_count += 1
                         except Exception as e:
                              st.error(f"Error processing update for QN {qn} (Paper {paper}): {e}")
                              update_fail_count += 1
                 
                 st.success("Supabase update process finished.")
                 st.write(f"- Successfully updated: {update_success_count}")
                 st.write(f"- Failed due to update/processing errors: {update_fail_count - json_parse_errors}")
                 st.write(f"- Failed due to invalid edited JSON: {json_parse_errors}")
                 # Optionally clear session state after update
                 # del st.session_state.processed_oeq_data
             else:
                  st.warning("No edited data found.")

# --- Run the app ---
if __name__ == "__main__":
     # Add authentication check if needed, similar to other pages
    if st.session_state.get("authenticated", False):
         main()
    else:
         st.warning("Please log in first.") 
