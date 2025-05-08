import streamlit as st
import pandas as pd
import os
from PyPDF2 import PdfReader # Keep PdfReader for page dimensions
from io import BytesIO
import tempfile
from PyPDFForm import PdfWrapper, FormWrapper
import re
#import fitz  # PyMuPDF - No longer needed here
#from openai import OpenAI - No longer needed here
import json # Need json

# Initialize session state variables (keep as is)
if 'processed_pdf' not in st.session_state:
    st.session_state.processed_pdf = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Remove the AI-based extraction function
# def extract_question_numbers_from_page(pdf_doc, page_num):
#    ...

def process_pdf(extracted_data_df, paper_code, tmp_pdf_path):
    """
    Process the PDF using extracted data, adding filled form fields.
    Returns the processed PDF bytes.
    """
    # Get page dimensions from the PDF using PdfReader
    try:
        with open(tmp_pdf_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            # Assuming all pages have the same dimensions as the first page
            first_page = pdf_reader.pages[0]
            mediabox = first_page.mediabox
            llx, lly = float(mediabox.lower_left[0]), float(mediabox.lower_left[1])
            urx, ury = float(mediabox.upper_right[0]), float(mediabox.upper_right[1])
            page_width = urx - llx
            page_height = ury - lly
    except Exception as e:
        st.error(f"Error reading PDF dimensions: {e}")
        return None

    # Open the PDF using PdfWrapper for modifications
    try:
        pdf_form = PdfWrapper(tmp_pdf_path)
    except Exception as e:
        st.error(f"Error opening PDF with PdfWrapper: {e}")
        return None

    # Dictionary to track the next Y coordinate for OEQs on each page
    page_y_coords = {}
    default_oeq_start_y = ury - 100 # Start 100 points from the top
    oeq_spacing = 70 # Spacing between OEQ fields

    # Dictionary to track the next Y coordinate for MCQs on each page (vertical stacking)
    page_mcq_y_coords = {}
    default_mcq_start_y = lly + 150 # Start higher to stack down
    mcq_fixed_x = urx - 80 # Fixed X near right margin
    mcq_height = 20
    mcq_spacing_y = mcq_height + 10 # Height + 10 points spacing vertically
    mcq_width = 30 # Keep width definition

    # Process each question from the DataFrame
    total_questions = len(extracted_data_df)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Dictionary to hold data for the .fill() method
    fill_data = {}

    for index, row in extracted_data_df.iterrows():
        try:
            page_number = int(row['page_number'])
            question_number = str(row['question_number'])
            question_type = str(row['question_type']).lower() # Ensure lowercase for comparison
            model_answer_json = row['model_answer']
            # Safely get marks, defaulting to 0 if NaN/None
            marks_val = row['marks_allocated']
            fmarks = 0 if pd.isna(marks_val) else int(marks_val)

            status_text.text(f"Processing question {index + 1}/{total_questions}: {question_number} on page {page_number}")

            # Construct field name
            question_label = f"{paper_code}_{question_number}_{fmarks}"

            # Determine the value to fill
            answer_value = "" # Default to empty string
            if question_type == "multiple_choice":
                answer_value = "" # Explicitly empty for MCQ
            else:
                # For other types, use the raw JSON
                answer_value = model_answer_json 
            
            # Add data to the dictionary for later filling
            fill_data[question_label] = answer_value
            #st.write(f"Queueing: {question_label} = {answer_value}") # Debug print

            # --- Create Widget (without value) --- 
            if question_type == "multiple_choice":
                 # Determine Y coordinate for MCQ
                 if page_number not in page_mcq_y_coords:
                     page_mcq_y_coords[page_number] = default_mcq_start_y
                 current_y = page_mcq_y_coords[page_number]

                 # Check if MCQ field goes below the bottom margin (approx)
                 if current_y < lly + 20: # Leave 20 points margin at bottom
                     st.warning(f"Too many MCQs stacked vertically on page {page_number}, potential overlap/overflow near bottom for Q {question_number}.")
                     # Reset Y to start for safety, though they will overlap the first ones
                     current_y = default_mcq_start_y 
                     # Consider alternative handling like starting a new column if needed

                 pdf_form.create_widget(
                     widget_type="text",
                     name=question_label,
                     page_number=page_number, 
                     x=mcq_fixed_x,  # Use fixed X
                     y=current_y, # Use calculated Y
                     width=mcq_width, # Use defined width
                     height=mcq_height,
                     max_length=5,
                     # value="", # REMOVED value parameter
                     font="Courier",
                     font_size=12, 
                     font_color=(0, 0, 0),
                     bg_color=(1, 1, 1), 
                     border_color=(1, 0, 0), 
                     border_width=1,
                     alignment=1, 
                     multiline=False 
                 )
                 # Update Y coordinate for the next MCQ on this page (moving *down*) 
                 page_mcq_y_coords[page_number] = current_y - mcq_spacing_y
            
            elif question_type in ["explanation", "experimental_purpose", "fact_recall"]:
                 # Determine Y coordinate for OEQ
                 if page_number not in page_y_coords:
                     page_y_coords[page_number] = default_oeq_start_y
                 current_y = page_y_coords[page_number]

                 # Ensure Y coordinate doesn't go off the page
                 if current_y < lly + 50: # Keep some margin from bottom
                     st.warning(f"Too many OEQs on page {page_number}, potential overlap for Q {question_number}. Adjusting position slightly.")
                     current_y = lly + 50

                 pdf_form.create_widget(
                     widget_type="text",
                     name=question_label,
                     page_number=page_number,
                     x=llx + 50, # 50 points from left margin
                     y=current_y,
                     width=page_width - 100, # Use available width with margins
                     height=50,
                     # value=answer_value, # REMOVED value parameter
                     font="Courier",
                     font_size=10,
                     font_color=(0, 0, 0),
                     bg_color=(0.9, 0.9, 0.9), # Keep gray background for distinction
                     border_color=(0, 0, 1),
                     border_width=1,
                     alignment=0, # Left alignment
                     multiline=True
                 )
                
                 page_y_coords[page_number] = current_y - oeq_spacing
            else:
                 # Handle unknown type if necessary, maybe add a default field
                 st.warning(f"Unknown question type '{question_type}' for question {question_number}. Creating field anyway.")
                 # Default widget creation logic (e.g., similar to OEQ)
                 if page_number not in page_y_coords:
                     page_y_coords[page_number] = default_oeq_start_y
                 current_y = page_y_coords[page_number]
                 if current_y < lly + 50:
                     current_y = lly + 50
                 pdf_form.create_widget(
                     widget_type="text",
                     name=question_label,
                     page_number=page_number,
                     x=llx + 50,
                     y=current_y,
                     width=page_width - 100,
                     height=50,
                     # value=answer_value, # REMOVED value parameter
                     font="Courier", font_size=10, font_color=(0, 0, 0),
                     bg_color=(0.9, 0.9, 0.9), border_color=(0, 0, 1), border_width=1,
                     alignment=0, multiline=True
                 )
                 page_y_coords[page_number] = current_y - oeq_spacing

        except Exception as e:
            st.error(f"Error processing question {row.get('question_number', 'N/A')}: {e}")
            continue # Skip to the next question
        finally:
             progress_bar.progress((index + 1) / total_questions)

    status_text.text("All widgets created. Saving intermediate PDF...")

    # --- Step 1: Save PDF with empty widgets --- 
    intermediate_pdf_bytes = None
    try:
        # Add paper code *before* reading bytes for intermediate file
        pdf_form.draw_text(
            text=paper_code,
            page_number=1,
            x=10,
            y=10,
            font_size=6,
        )
        intermediate_pdf_bytes = pdf_form.read()
    except Exception as e:
        st.error(f"Error saving PDF with created widgets: {e}")
        return None # Cannot proceed if saving fails
    
    # --- Step 2: Create intermediate temp file and fill using FormWrapper --- 
    final_pdf_bytes = None
    intermediate_tmp_pdf_path = None # Define variable scope
    try:
        # Create a second temporary file for the intermediate state
        with tempfile.NamedTemporaryFile(suffix="_intermediate.pdf", delete=False) as intermediate_tmp_pdf:
            intermediate_tmp_pdf.write(intermediate_pdf_bytes)
            intermediate_tmp_pdf_path = intermediate_tmp_pdf.name

        status_text.text("Intermediate PDF saved. Filling data...")
        # Now use FormWrapper on the intermediate PDF which contains the fields
        pdf_filler = FormWrapper(intermediate_tmp_pdf_path)
        pdf_filler.fill(fill_data, adobe_mode=True) # Use the fill_data dict
        
        status_text.text("Data filled. Reading final PDF...")
        # Read the final bytes from the filled PDF
        final_pdf_bytes = pdf_filler.read()
        status_text.text("PDF Finalized!")

    except Exception as e:
        st.error(f"Error during data filling or final read: {e}")
        # Keep final_pdf_bytes as None
    finally:
        # Clean up the *intermediate* temporary file
        if intermediate_tmp_pdf_path and os.path.exists(intermediate_tmp_pdf_path):
            os.unlink(intermediate_tmp_pdf_path)

    # Return the final bytes (might be None if error occurred)
    return final_pdf_bytes

def main():
    st.title("Create SA2 Paper with Form Fields & Answers")

    # Check if extracted data exists in session state
    if "extracted_data" not in st.session_state or st.session_state.extracted_data is None:
        st.error("Extracted question data not found. Please go to 'Upload and Extract' page first.")
        st.stop()

    extracted_data_df = st.session_state.extracted_data

    # Display the extracted data for confirmation (optional)
    st.subheader("Extracted Questions and Answers")
    st.dataframe(extracted_data_df)

    # Step 1: User Inputs (from 6Upload_MCQ_answers.py)
    level = st.selectbox("Select Primary Level", ["P3", "P4", "P5", "P6"])
    year = st.text_input("Year")
    school = st.selectbox("Select School", [
        "ACSP", "ATPS", "HPPS", "MBPS", "MGSP", "NHPS", "ACSJ",
        "NYPS", "RGPS", "SCGS", "SHPS", "SJIJ", "TNPS", "RSPS",
        "CHSP", "PCPS", "RSSP"
    ])
    paper_type = st.text_input("Type of Paper, WA2, SA1 etc")

    paper_code = f"{level}{year}{school}{paper_type}"
    st.write(f"Paper Code: {paper_code}")

    # Step 2: Upload PDF
    pdf_file = st.file_uploader("Upload the *same* PDF Document used for extraction", type=["pdf"])
    
    if pdf_file:
        # Read the PDF bytes once
        pdf_bytes = pdf_file.read()
        
        # Create a temporary file to work with
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_pdf.write(pdf_bytes)
            tmp_pdf_path = tmp_pdf.name

        # Add a button to start processing
        if st.button("Create PDF with Filled Answers"):
            try:
                # Process the PDF using the DataFrame
                processed_pdf = process_pdf(extracted_data_df, paper_code, tmp_pdf_path)

                # Show download button if processing is complete
                if processed_pdf:
                    st.download_button(
                        "Download PDF with Answers",
                        data=processed_pdf,
                        file_name=f"{paper_code}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("Failed to process PDF.")
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
            finally:
                # Clean up temporary file
                if 'tmp_pdf_path' in locals() and os.path.exists(tmp_pdf_path):
                    os.unlink(tmp_pdf_path)

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    main()
else:
    st.warning("Please log in first before using SM AI-Tutor.") 