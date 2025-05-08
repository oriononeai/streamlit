import streamlit as st
import pymupdf
from openai import OpenAI
import os
import json
import pandas as pd

# --- Authentication Check --- 
if st.session_state.get("authenticated", False):

    st.title("Upload PDF and Submit to AI")

    # --- User Inputs for Paper Details (using dropdowns) ---
    st.subheader("Select Paper Details")

    # Define options
    level_options = ["--Select--", "P3", "P4", "P5", "P6"]
    school_options = ["--Select--", "ACSP", "ATPS", "HPPS", "MBPS", "MGSP", "NHPS", "ACSJ", "NYPS", "RGPS", "SCGS", "SHPS", "SJIJ", "TNPS", "RSPS", "CHSP", "PCPS", "RSSP"]
    year_options = ["--Select--"] + [str(y) for y in range(2027,2019, -1)] # Example: 2024 down to 2020

    # Create select boxes
    level = st.selectbox("Level", options=level_options, key="level_select")
    school = st.selectbox("School", options=school_options, key="school_select")
    exam_type = st.text_input("Type of Paper, WA2, SA1 etc")
    year = st.selectbox("Year", options=year_options, key="year_select")

    st.subheader("Upload PDF File")
    uploaded_file = st.file_uploader("Upload your Science Paper (.pdf)", type=["pdf"], key="pdf_uploader")
    mcq_answer_page = st.number_input("MCQ Answer Key Page Number (Optional)", min_value=1, step=1, value=None, key="mcq_page_input", help="If the PDF has a separate page listing MCQ answers, enter its page number here.")
    mcq_marks_override = st.number_input("Set Fixed Marks for All MCQ Questions (Optional)", min_value=0, step=1, value=None, key="mcq_marks_input", help="If set, this mark value will override any extracted marks for MCQ questions.")
    # ------------------------------------------------------

    # Create OpenAI client using environment variable
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Function to extract text from uploaded PDF
    def extract_text_from_pdf(uploaded_file):
        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")
        full_text = ""
        for i, page in enumerate(doc):
            full_text += f"\n--- PAGE {i + 1} ---\n"
            full_text += page.get_text()
        return full_text

    def safe_extract_json(text):
        """
        Extract JSON array from first `[` to last `]` for messy ChatGPT response.
        Works when response is supposed to be one big JSON array.
        """
        try:
            start_idx = text.index('[')
            end_idx = text.rindex(']')
            json_text = text[start_idx:end_idx+1]
            return json.loads(json_text)
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to extract valid JSON: {e}")

    # System prompt to tell ChatGPT how to structure the paper
    system_prompt = """
    You are a Science exam paper processing assistant specializing in model answer generation.

    You will receive text extracted from a Primary School Science exam paper.
    The text includes markers like '--- PAGE X ---' to indicate the start of each page.
    Optionally, you may also receive a separate block of text labeled '--- MCQ Answer Key Text ---' extracted from a specific page containing the answers to multiple-choice questions.

    Your primary task is to:
    1.  Extract structure for each individual question part (page, number, text, type, marks).
    2.  **Generate a concise and scientifically accurate model answer string** for each question based on Primary School Science knowledge, **prioritizing the provided MCQ Answer Key Text for multiple_choice questions if available.**

    ---

    For every question part:

    1.  Identify the page number it belongs to based on the most recent '--- PAGE X ---' marker preceding the question text.
    2.  Extract the following fields:
        - `page_number`: The page number (integer) where the question starts.
        - `question_number`: Include subparts like (a), (b), etc. Example: "40a", "40b", "1", "1a"
        - `full_question_text`: Copy the full question text for that part only.
        - `question_type`: Choose one:
            - "explanation" — if the question asks for reasons, causes, scientific processes, effects
            - "experimental_purpose" — if the question asks about the aim of an experiment, controlled variables, fair test conditions
            - "fact_recall" — if the question requires stating a simple fact, definition, or direct knowledge
            - "range" - if the answer to the question is between a range of numbers, for example "between 1 and 5"
            - "multiple_choice" — if the question provides options numbered (1), (2), (3), (4)
        - `marks_allocated`: Extract from the number inside square brackets `[ ]`. Example: `[2]` becomes `marks_allocated: 2`. If no marks shown, set it as `null`.
        - `model_answer`: Generate the model answer content as a single string. 
            - **For multiple_choice questions:** If the '--- MCQ Answer Key Text ---' is provided, find the answer corresponding to the `question_number` in that text. Use that number ("1", "2", "3", or "4") as the `model_answer` string. If the key text is not provided or the answer is not found there, predict the answer based on the question text and provide only the option number as the string.
            - **For range questions:** Provide the range as a string (e.g., "1-5"). 
            - **For explanation questions:** Provide the explanation as a concise string.
            - **For fact_recall questions:** Provide the factual answer or definition as a concise string.
            - **For experimental_purpose questions:** Provide the answer regarding the aim, variable, or condition as a concise string.

    ---

    Model Answer Generation Rules:
    - Generate the most appropriate and accurate model answer **as a single string** based on the question text and standard Primary School Science curriculum.
    - **Prioritize the MCQ Answer Key Text (if provided) for determining multiple_choice answers.**
    - Be concise.
    - For multiple choice questions where the answer key is not used or applicable, the string should only contain the correct option number.
    - If the question is ambiguous, relies heavily on a diagram not fully represented in the text, or requires external knowledge beyond typical Primary School Science, make the best possible attempt. If a reasonable answer string cannot be generated, leave the `model_answer` field as an empty string (`""`).

    ---

    Handling of Tables, Diagrams, Images:
    - Ignore purely decorative diagrams, pictures, illustrations, and non-text images.
    - **If a table contains multiple-choice options, experimental data, observations, or essential information, you must read and process the table to generate the correct model answer string.**
    - Use any text extracted from tables to help generate the answer string.

    ---

    Formatting Instructions:
    - Output only a pure JSON array containing all extracted and generated question structures.
    - Do NOT add any extra explanation, notes, comments, headings, or markdown (such as ```json).

    ---

    Example of Output:

    [
      {
        "page_number": 10,
        "question_number": "40a",
        "full_question_text": "Based on Graph 1, which spring, P or Q, can be stretched more easily? Give a reason for your answer.",
        "question_type": "explanation",
        "marks_allocated": 1,
        "model_answer": "Spring P. It shows a greater extension for the same applied force compared to Spring Q."
      },
      {
        "page_number": 1,
        "question_number": "1",
        "full_question_text": "Which of these animals reproduces by laying eggs? (1) Bat (2) Whale (3) Platypus (4) Cat",
        "question_type": "multiple_choice",
        "marks_allocated": 1,
        "model_answer": "3" // This might come from the Answer Key Text if provided and found
      }
    ]
    """

    # Check if all required inputs have valid selections
    all_inputs_valid = (
        uploaded_file is not None and
        level != "--Select--" and
        school != "--Select--" and
        exam_type != "--Select--" and
        year != "--Select--"
    )
    paper_code = f"{level}{year}{school}{exam_type}"
    st.write(f"Paper Code: {paper_code}")

    if all_inputs_valid:
        # This block runs only if valid selections are made and file is uploaded
        if st.button("Submit to AI for Structuring"):
            # Construct the paper code from selections        
            st.info(f"Processing paper with code: {paper_code}")

            # Process the PDF to get main text and optionally MCQ key text
            mcq_answer_key_text = None
            full_text = ""
            doc = None # Initialize doc to None
            try:
                # Ensure the file pointer is at the beginning
                uploaded_file.seek(0)
                doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")

                # 1. Extract main text from all pages
                for i, page in enumerate(doc):
                    full_text += f"\n--- PAGE {i + 1} ---\n"
                    full_text += page.get_text()

                # 2. Extract MCQ answer key text if page number is provided
                if mcq_answer_page is not None:
                    page_idx = int(mcq_answer_page) - 1 # Convert to 0-based index
                    if 0 <= page_idx < doc.page_count:
                        try:
                            mcq_answer_key_text = doc[page_idx].get_text()
                            st.success(f"Extracted text from MCQ answer key page {mcq_answer_page}.")
                            # Optional: Show the extracted key text
                            # st.text_area(f"MCQ Answer Key Text (Page {mcq_answer_page})", mcq_answer_key_text, height=150)
                        except Exception as page_err:
                            st.warning(f"Could not extract text from page {mcq_answer_page}: {page_err}")
                    else:
                        st.warning(f"Invalid page number ({mcq_answer_page}) for MCQ answer key. Document has {doc.page_count} pages.")

            except Exception as pdf_err:
                st.error(f"Error processing PDF file: {pdf_err}")
                st.stop()
            finally:
                 if doc: # Ensure doc exists before trying to close
                    doc.close()

            # If PDF processing failed, full_text might be empty
            if not full_text:
                st.error("Failed to extract any text from the PDF.")
                st.stop()

            # st.success("PDF extracted successfully!") # Moved success message earlier or remove
            # st.text_area("Extracted Text", full_text, height=300)

            # Construct user message for AI
            user_message_content = full_text
            if mcq_answer_key_text:
                user_message_content += f"\n\n--- MCQ Answer Key Text (Page {mcq_answer_page}) ---\n{mcq_answer_key_text}"

            with st.spinner("Processing with AI..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message_content} # Pass combined message
                        ],
                        temperature=0
                    )
                    structured_output = response.choices[0].message.content
                except Exception as e:
                    st.error(f"Error calling OpenAI API: {e}")
                    st.stop()

                try:
                    structured_data = safe_extract_json(structured_output)
                    st.success("AI structuring completed!")
                    # Build the DataFrame manually with new column names and MCQ marks override
                    data = []
                    for item in structured_data:
                        q_type = item.get("question_type")
                        ai_marks = item.get("marks_allocated")
                        
                        # Determine the final marks
                        final_marks = ai_marks # Default to AI extracted marks
                        if q_type == "multiple_choice" and mcq_marks_override is not None:
                            final_marks = mcq_marks_override # Override if MCQ and value is set
                        
                        row = {
                            "paper": paper_code,
                            "question_number": item.get("question_number"),
                            "level": level,
                            "question": item.get("full_question_text"),
                            "answer": item.get("model_answer"),
                            "marks": final_marks, # Use the determined final marks
                            "question_type": q_type,
                            "page_number": item.get("page_number")
                        }
                        data.append(row)
                    df = pd.DataFrame(data)
                    # Reorder columns if needed (optional, but good practice)
                    desired_columns = ["paper", "level", "question_number", "page_number", "question", "answer", "marks", "question_type"]
                    # Filter out any columns that might not exist (though they should)
                    existing_columns = [col for col in desired_columns if col in df.columns]
                    df = df[existing_columns]
                    
                    # Clean the question_number column
                    if 'question_number' in df.columns:
                        # Ensure the column is string type
                        df['question_number'] = df['question_number'].astype(str)
                        # Remove specified characters using regex
                        df['question_number'] = df['question_number'].str.replace(r'[()*\s]', '', regex=True)
                        # Optional: Convert back to original type if needed, but string is likely fine
                        
                    st.dataframe(df)
                    # Store the DataFrame with new structure in session state
                    st.session_state.extracted_data = df

                except ValueError as e:
                    st.error(f"Error extracting JSON from AI response: {e}")
                except Exception as e:
                     st.error(f"An error occurred processing the AI response: {e}")

    elif not all_inputs_valid and st.button("Submit to AI for Structuring"):
        # Show warning if button pressed without valid selections
        st.warning("Please select Level, Subject, Exam Type, Year, and upload a PDF file first.")

else:
    st.warning("Please log in first to upload and extract papers.")

