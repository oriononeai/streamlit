import streamlit as st
import pandas as pd
import re
import os
from supabase import create_client, Client

# Initialize session state
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'input_fields_metadata' not in st.session_state:
    st.session_state.input_fields_metadata = []

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

def fetch_answer_from_supabase(paper: str, question_number: str) -> str:
    """
    Fetches the answer from the Supabase table `pri_sci_paper` based on paper and question_number.
    Returns the answer if found, otherwise returns an empty string.
    """
    try:
        response = supabase.table("pri_sci_paper") \
            .select("answer") \
            .eq("paper", paper) \
            .eq("question_number", question_number) \
            .limit(1) \
            .execute()

        if response.data and len(response.data) > 0:
            return response.data[0].get("answer", "")
        else:
            return ""
    except Exception as e:
        print(f"Error fetching answer: {e}")
        return ""


def parse_placeholder_attributes(placeholder_str):
    # Expects a string like "[input type=text question_number=1 ...]"
    # Regex ensures it starts with "[input ", has attributes, and ends with "]"
    match = re.match(r'^\[input\s(.*?)\]$', placeholder_str) # Ensure space after "input"
    if not match:
        return None
    
    attrs_str = match.group(1)
    attributes = {}
    # Split attributes by space. Handles multiple spaces between attributes.
    parts = [p for p in attrs_str.split(" ") if p] 
    for part in parts:
        key_value = part.split("=", 1)
        if len(key_value) == 2:
            attributes[key_value[0]] = key_value[1]
    return attributes

def extract_marks(text_block):
    """Extracts the number from the last found [number] pattern in the text block."""
    matches = re.findall(r'\s*\[(\d+)\]', text_block) # Finds [1], [2], etc. allowing optional leading space
    if matches:
        return int(matches[-1]) # Return the last one found as an integer
    return None # Or 0, or other default if no marks found

def render_markdown_with_st_image(markdown_lines, md_file_dir):
    """Renders markdown lines, intercepting image tags to use st.image."""
    # Regex to find Markdown image tags: !\[alt_text\](image_path)
    # It captures: 1=alt_text, 2=image_path
    img_tag_re = re.compile(r'!\[(.*?)\]\((.*?)\)')
    
    current_text_segment = []

    for line in markdown_lines:
        match_found_on_line = False
        last_idx = 0
        for match in img_tag_re.finditer(line):
            match_found_on_line = True
            start, end = match.span()
            
            # Add text before the image tag to the current segment
            current_text_segment.append(line[last_idx:start])
            if current_text_segment:
                st.markdown("\n".join(current_text_segment), unsafe_allow_html=False) # Keep unsafe_allow_html=False for now
                current_text_segment = []
            
            alt_text = match.group(1)
            image_path = match.group(2)
            
            # Attempt to display image using st.image - paths should be like "media/folder/img.png"
            # md_file_dir is CWD (".") in our case
            full_image_path_for_st_image = os.path.join(md_file_dir, image_path)

            if os.path.exists(full_image_path_for_st_image):
                st.image(full_image_path_for_st_image, caption=alt_text if alt_text else None)
            else:
                st.warning(f"Image not found by st.image: {full_image_path_for_st_image}. Original path: {image_path}")
                # Fallback: render the original markdown tag if st.image can't find it, hoping browser might
                st.markdown(match.group(0), unsafe_allow_html=False)

            last_idx = end
        
        # Add any remaining text after the last image tag (or the whole line if no image)
        if match_found_on_line:
            current_text_segment.append(line[last_idx:])
        else:
            current_text_segment.append(line)
            
    # Render any final text segment
    if current_text_segment:
        st.markdown("\n".join(current_text_segment), unsafe_allow_html=False)

def build_ui_and_collect_metadata_revamped(md_content, md_file_dir, paper_no):
    st.session_state.input_fields_metadata = [] 
    all_lines = md_content.splitlines()
    current_md_block_lines_for_processing = []

    for line_idx, line_text in enumerate(all_lines):
        stripped_line_text = line_text.strip()
        attrs = None
        is_placeholder_line = False

        # Check if the stripped line is a placeholder
        if stripped_line_text.startswith("[input ") and stripped_line_text.endswith("]"):
            attrs = parse_placeholder_attributes(stripped_line_text)
            if attrs and 'question_number' in attrs:
                is_placeholder_line = True

        if is_placeholder_line:
            question_for_df = ""
            if current_md_block_lines_for_processing:
                question_for_df = "\n".join(current_md_block_lines_for_processing)
                render_markdown_with_st_image(current_md_block_lines_for_processing, md_file_dir)
                current_md_block_lines_for_processing = [] 

            # 2. Process and render the placeholder as a Streamlit widget
            q_num_attr = attrs['question_number']
            placeholder_type_from_attr = attrs.get('type', 'text')
            marks_for_df = extract_marks(question_for_df)
            widget_key = f"answer_{q_num_attr}_{line_idx}"

            # Store metadata for later retrieval
            st.session_state.input_fields_metadata.append({
                'widget_key': widget_key,
                'paper_no': paper_no,
                'question_number_df': q_num_attr, 
                'question_df': question_for_df,
                'marks_df': marks_for_df,
                'question_type_df': "",
                'attrs': attrs
            })
            
            # Provide a label for the input widget for clarity
            input_label = f"{q_num_attr}:"
            
            # Fetch existing answer from Supabase
            existing_answer = fetch_answer_from_supabase(paper_no, q_num_attr)

            if placeholder_type_from_attr == "text":
                st.text_input(input_label, key=widget_key, value=existing_answer, label_visibility="visible")
            elif placeholder_type_from_attr == "textarea":
                rows = int(attrs.get('rows', 3)) # Default to 3 rows
                # Ensure minimum rows for height calculation if multiplier is 25
                # 68/25 = 2.72, so effectively need at least 3 rows
                actual_rows_for_height = max(3, rows) 
                st.text_area(input_label, key=widget_key, height=actual_rows_for_height * 25, value=existing_answer, label_visibility="visible")
            elif placeholder_type_from_attr == "canvas": # For canvas, we show original image if src, then textarea
                if 'src' in attrs:
                    img_src_path = attrs['src']
                    # Image path resolution: try relative to md_file_dir, then CWD for media folder
                    full_img_path = os.path.join(md_file_dir, img_src_path) 
                    if os.path.exists(full_img_path):
                        st.image(full_img_path)
                    else:
                        # Fallback for common "media" folder relative to app CWD
                        alt_img_path = os.path.join(".", img_src_path) 
                        if os.path.exists(alt_img_path) and alt_img_path != full_img_path:
                            st.image(alt_img_path)
                        else:
                            st.warning(f"Canvas src image not found: {full_img_path}")
                rows = int(attrs.get('rows', 4)) # Default to 4 rows for canvas's textarea
                # Ensure minimum rows for height calculation
                actual_rows_for_height = max(3, rows) # Canvas text area also needs min 3 rows based on 68px/25px
                st.text_area(f"{q_num_attr}: (drawing/description)", key=widget_key, height=actual_rows_for_height * 25, value=existing_answer, label_visibility="visible")
            else:
                # If unknown, render the original placeholder line as Markdown text for visibility
                st.markdown(f"**Unknown placeholder type found:** `{line_text}`")
                # Also add it to current_md_block_lines_for_processing so it's not lost if it was a parsing mistake
                current_md_block_lines_for_processing.append(line_text)
        
        else: # Not a placeholder line
            current_md_block_lines_for_processing.append(line_text)

    # Render any remaining Markdown content at the end of the file
    if current_md_block_lines_for_processing:
        render_markdown_with_st_image(current_md_block_lines_for_processing, md_file_dir)

def main():
    st.title("Markdown Exam Answer Tool (Revamped)")

    # Step 1: Select Level and Year
    level = st.selectbox("Select Level", ["P3", "P4", "P5", "P6"])
    year = st.selectbox("Select Year", [str(y) for y in range(2020, 2026)])

    # Step 2: Concatenate to form folder name
    folder_name = f"{level}{year}"

    # Step 3: Retrieve files from Supabase storage
    bucket_name = "markdown"
    response = supabase.storage.from_(bucket_name).list(path=folder_name)

    # Step 4: Extract file names
    if response:
        file_names = [file['name'] for file in response if file['name'].endswith('.md')]
        selected_file = st.selectbox("Select a Markdown File", file_names)
    else:
        st.warning("No files found in the selected folder.")
        selected_file = None

    # Full path to the file in the bucket
    file_path = f"{folder_name}/{selected_file}"

    # Download the file content
    file_response = supabase.storage.from_(bucket_name).download(file_path)

    # Decode the content (assuming it's UTF-8 encoded Markdown)
    if file_response:
        markdown_content = file_response.decode("utf-8")
        st.markdown(markdown_content)
    else:
        st.error("Failed to download the selected file.")

    uploaded_md_file = markdown_content

    if uploaded_md_file is not None:
        md_content = uploaded_md_file.getvalue().decode("utf-8")
        
        # Extract paper_no from filename, remove .md extension
        paper_name_full = uploaded_md_file.name
        paper_no = os.path.splitext(paper_name_full)[0]

        # md_file_dir will be used for relative image paths if any
        # For uploaded files, there isn't a true "directory" unless it's a zip
        # Using "." means it assumes images (e.g., in a 'media' folder) are relative to CWD of Streamlit app
        md_file_dir = "." 

        build_ui_and_collect_metadata_revamped(md_content, md_file_dir, paper_no)

        if st.button("Save Answers"):
            data_for_df = []
            if not st.session_state.input_fields_metadata:
                st.warning("No input fields were processed to save.")
            for field_meta in st.session_state.input_fields_metadata:
                answer = st.session_state.get(field_meta['widget_key'], "")
                level = field_meta['paper_no'][:2] if field_meta['paper_no'] else None
                
                raw_question = field_meta['question_df']
                cleaned_lines = []
                for line in raw_question.splitlines():
                    # 1. Remove header lines like '## Qxx'
                    if re.match(r"^##\s*Q\d+", line.strip()):
                        continue # Skip this line
                    # 2. Remove '#tag' from question lines
                    cleaned_line = re.sub(r'\s*#\w+', '', line)
                    cleaned_lines.append(cleaned_line)
                cleaned_question = "\n".join(cleaned_lines).strip() # .strip() to remove leading/trailing newlines from overall block

                data_for_df.append({
                    "level": level,
                    "paper": field_meta['paper_no'],
                    "question_number": field_meta['question_number_df'],
                    "question": cleaned_question, # Use cleaned text 
                    "answer": answer,
                    "marks": field_meta['marks_df'],
                    "question_type": field_meta['question_type_df']
                })
            if data_for_df:
                # Define column order for consistency
                column_order = ["level", "paper", "question_number", "question", "answer", "marks", "question_type"]
                df = pd.DataFrame(data_for_df)
                # Reorder columns if not all columns are present (e.g. if some are None like marks initially)
                # This ensures that if a column is all None, it still appears if specified in column_order
                df = df.reindex(columns=column_order)
                st.session_state.extracted_data = df
                st.success("Answers extracted successfully!")
                st.dataframe(st.session_state.extracted_data)
            elif st.session_state.input_fields_metadata: # If metadata exists but no data_for_df (e.g. all answers empty)
                st.info("Answers saved (all were empty).") 
                st.session_state.extracted_data = pd.DataFrame(data_for_df) # Show empty DF
                st.dataframe(st.session_state.extracted_data)
            # else: (No metadata and no data_for_df) - handled by the first warning

    else:
        st.info("Please upload a Markdown file to begin.")

if __name__ == "__main__":
    main() 
