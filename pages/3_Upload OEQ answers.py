import os
import pandas as pd
import pypdf
# Removed pymupdf as it's not used
import streamlit as st
# Removed openai
import json # Keep for potential JSON handling if needed, otherwise remove later
# Removed asyncio
from supabase import create_client, Client

# --- Environment Selection ---
ENV_OPTIONS = ["QA", "PROD"]
# Place selectbox in sidebar for consistency if desired, or main area if not.
# Assuming sidebar like the other file for now.
selected_env = st.sidebar.selectbox("Select Environment", ENV_OPTIONS, index=0, key="oeq_env_select")

# --- Supabase Initialization ---
if selected_env == "QA":
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")
elif selected_env == "PROD":
    supabase_url = os.getenv("SUPABASEO1_URL")
    supabase_key = os.getenv("SUPABASEO1_API_KEY")
else: # Default to QA
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")
    st.sidebar.warning(f"Unknown environment '{selected_env}'. Defaulting to QA.")

supabase: Client | None = None
supabase_available = False

try:
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        supabase_available = True
        st.sidebar.success(f"Supabase connection established for {selected_env}.") # User feedback
    else:
        st.sidebar.warning(f"Supabase URL/Key missing for {selected_env}. Cannot query/save questions.")
except Exception as e:
    st.sidebar.error(f"Supabase connection failed: {e}")
# ---------------------------

def main():
    st.title("Upload and Edit Model Answers") # Changed title
    # Upload the PDF file
    pdf_file = st.file_uploader("Upload PDF with Model Answers in Form Fields", type=["pdf"], key="pdfform_simple")

    # --- Removed level input ---

    if pdf_file is None:
        st.info("Please upload a PDF file.")
        return

    # --- Removed OpenAI availability check ---

    # ----- Process Form Fields using pypdf ----- 
    st.write("=== Processing PDF Form Fields ===")
    pdf_file.seek(0)
    pdf_reader = pypdf.PdfReader(pdf_file)
    fields = pdf_reader.get_fields()

    # field_data will store dictionaries for each valid field found
    field_data = []

    if fields:
        st.write(f"Found {len(fields)} form fields. Extracting data...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        processed_field_names = set()

        for i, (field_name, field) in enumerate(fields.items()):
            # Skip duplicates
            if field_name in processed_field_names:
                continue
            processed_field_names.add(field_name)

            value = field.get('/V')
            raw_answer = value.strip() if isinstance(value, str) else None

            status_text.text(f"Processing field {i+1}/{len(fields)}: {field_name}")

            if raw_answer: # Only process fields with answers
                try:
                    # Format: paper_question number_marks optional_draw
                    field_parts = field_name.split('_')
                    if len(field_parts) >= 3:
                        paper = field_parts[0].strip()
                        qn_str = field_parts[1].strip()
                        marks_part = field_parts[2]
                        marks_details = marks_part.split()
                        is_draw_question = False
                        try:
                            marks = int(marks_details[0])
                            if len(marks_details) > 1 and marks_details[1].lower() == 'draw':
                                is_draw_question = True
                        except (ValueError, IndexError):
                            st.warning(f"Could not parse marks from '{marks_part}' for field '{field_name}'. Skipping.")
                            continue

                        # --- Fetch question text from Supabase --- 
                        question_text = "QUESTION TEXT NOT FOUND"
                        if supabase_available and supabase:
                            try:
                                query = supabase.table("pri_sci_paper")\
                                              .select("question")\
                                              .eq("paper", paper)\
                                              .eq("question_number", qn_str)\
                                              .limit(1)\
                                              .execute()
                                if query.data:
                                    question_text = query.data[0]['question']
                                else:
                                    st.warning(f"Could not find question in Supabase for Paper '{paper}', QN '{qn_str}'.")
                            except Exception as db_error:
                                st.error(f"Supabase query failed for Paper '{paper}', QN '{qn_str}': {db_error}")
                                question_text = "SUPABASE QUERY ERROR"
                        else:
                             st.warning("Supabase not available. Cannot fetch question text.")
                             question_text = "SUPABASE UNAVAILABLE"

                        # Store extracted data for this field
                        row_data = {
                            "paper": paper,
                            "question_number": qn_str,
                            "question": question_text,
                            "marks": marks,
                            "answer": raw_answer, # Raw model answer from PDF
                            "is_draw": is_draw_question
                        }
                        field_data.append(row_data)

                    else:
                        st.warning(f"Field name '{field_name}' format invalid. Skipping.")
                except Exception as e:
                    st.error(f"Error processing field '{field_name}': {e}")
            # Update progress bar
            progress = (i + 1) / len(fields)
            progress_bar.progress(min(progress, 1.0))

        progress_bar.empty()
        status_text.empty()

        if field_data:
            st.success("Extracted data from PDF. You can edit the 'marks' and 'answer' columns below.")
            # Create DataFrame directly from the collected data
            df_processed = pd.DataFrame(field_data)

            # --- Display Editable DataFrame --- 
            # No need for session state to store original, update will use editor's state
            edited_df = st.data_editor(
                 df_processed,
                 num_rows="fixed",
                 column_config={
                     "question": st.column_config.TextColumn("Question Text", disabled=True),
                     "paper": st.column_config.TextColumn("Paper", disabled=True),
                     "question_number": st.column_config.TextColumn("QN", disabled=True),
                     "answer": st.column_config.TextColumn("Model Answer (Editable)"),
                     "marks": st.column_config.NumberColumn("Marks (Editable)", min_value=0, step=1),
                     "is_draw": st.column_config.CheckboxColumn("Is Draw Question?", disabled=True),
                 },
                 disabled=["paper", "question_number", "question", "is_draw"], # Explicitly disable non-editable cols
                 height=500, # Adjust height
                 key="simple_answers_editor"
            )

            # --- Add Update Button --- 
            if supabase_available and supabase:
                if st.button("Update Supabase with Edited Answers"):
                    if edited_df is not None:
                        updates_made = 0
                        update_errors = 0
                        with st.spinner("Updating Supabase..."):
                            # Iterate through the potentially edited DataFrame
                            for index, edited_row in edited_df.iterrows():
                                try:
                                    # Prepare payload with only the columns to update
                                    update_payload = {
                                        "marks": edited_row['marks'],
                                        "answer": edited_row['answer'] # Map df 'answer' to supabase 'answer'
                                    }
                                    # Keys for matching the row
                                    match_keys = {
                                        'paper': edited_row['paper'],
                                        'question_number': edited_row['question_number']
                                    }
                                    
                                    # Perform the update
                                    response = supabase.table("pri_sci_paper")\
                                                  .update(update_payload)\
                                                  .match(match_keys)\
                                                  .execute()
                                    
                                    # Basic check for errors (Supabase client might not populate .data on update)
                                    if hasattr(response, 'error') and response.error:
                                         st.error(f"Error updating QN {match_keys['question_number']} (Paper {match_keys['paper']}): {response.error.message}")
                                         update_errors += 1
                                    # elif response.data: # Check if data exists and indicates success, depends on API version
                                    #     updates_made += len(response.data) 
                                    else: # Assume success if no error reported
                                        updates_made += 1 # Increment count assuming update worked if no error

                                except Exception as update_e:
                                    st.error(f"Exception updating QN {edited_row['question_number']} (Paper {edited_row['paper']}): {update_e}")
                                    update_errors += 1

                        if update_errors == 0:
                            st.success(f"Successfully processed updates for {updates_made} records in Supabase!")
                        else:
                            st.warning(f"Completed update process with {update_errors} errors for {updates_made} potential successes.")
                    else:
                         st.warning("Edited data is missing. Cannot update.")
            elif not supabase_available:
                 st.warning("Supabase is not configured. Cannot save updates.")

        else:
            st.write("No valid form fields containing answers found or processed successfully.")
    else:
        st.write("No form fields found in the PDF.")

# Keep authentication check
if st.session_state.get("authenticated", False):
    main()
else:
    st.warning("Please log in first before using SM AI-Tutor.")
