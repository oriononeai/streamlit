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


def render_markdown_with_st_image(markdown_lines, supabase_client, bucket_name, base_folder):
    """Renders markdown lines, intercepting image tags to fetch from Supabase storage."""
    # Regex to find Markdown image tags: !\[alt_text\](image_path)
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
                st.markdown("\n".join(current_text_segment), unsafe_allow_html=False)
                current_text_segment = []

            alt_text = match.group(1)
            image_path = match.group(2)  # e.g., "media/P62024ASCJPL/1i.png"

            # Split the path at the first slash
            parts = image_path.split('/', 1)
            
            if len(parts) == 2:
                bucket = parts[0]      # 'media'
                download_path = parts[1]  # 'P62024ASCJPL/1i.png'                
                

            # Fetch image from Supabase storage
            try:
                # Now you can use these variables in your Supabase client
                image_response = supabase_client.storage.from_(bucket).download(download_path)

                if image_response:
                    # Display image using st.image with the binary data
                    st.image(image_response, caption=alt_text if alt_text else None)
                else:
                    st.warning(f"Image not found in storage: {image_path}")
                    # Fallback: render the original markdown tag
                    st.markdown(match.group(0), unsafe_allow_html=False)

            except Exception as e:
                st.warning(f"Error fetching image {image_path}: {str(e)}")
                # Fallback: render the original markdown tag
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

def build_ui_and_collect_metadata_revamped(md_content, supabase_client, bucket_name, base_folder, paper_no):
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
                render_markdown_with_st_image(current_md_block_lines_for_processing, supabase_client, bucket_name,
                                              base_folder)
                current_md_block_lines_for_processing = []

                # Process and render the placeholder as a Streamlit widget
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
                rows = int(attrs.get('rows', 3))
                actual_rows_for_height = max(3, rows)
                st.text_area(input_label, key=widget_key, height=actual_rows_for_height * 25, value=existing_answer,
                             label_visibility="visible")
            elif placeholder_type_from_attr == "canvas":
                if 'src' in attrs:
                    img_src_path = attrs['src']  # e.g., "media/P62024ASCJPL/1i.png"

                    # Fetch image from Supabase storage
                    try:
                        image_response = supabase_client.storage.from_(bucket_name).download(img_src_path)
                        if image_response:
                            st.image(image_response)
                        else:
                            st.warning(f"Canvas src image not found in storage: {img_src_path}")
                    except Exception as e:
                        st.warning(f"Error fetching canvas image {img_src_path}: {str(e)}")

                rows = int(attrs.get('rows', 4))
                actual_rows_for_height = max(3, rows)
                st.text_area(f"{q_num_attr}: (drawing/description)", key=widget_key, height=actual_rows_for_height * 25,
                             value=existing_answer, label_visibility="visible")
            else:
                st.markdown(f"**Unknown placeholder type found:** `{line_text}`")
                current_md_block_lines_for_processing.append(line_text)

        else:
            current_md_block_lines_for_processing.append(line_text)

    # Render any remaining Markdown content at the end of the file
    if current_md_block_lines_for_processing:
        render_markdown_with_st_image(current_md_block_lines_for_processing, supabase_client, bucket_name, base_folder)


def main():
    st.title("Markdown Exam Answer Tool (Revamped)")

    # Step 1: Select Level and Year
    level = st.selectbox("Select Level", ["P3", "P4", "P5", "P6"])
    year = st.selectbox("Select Year", [str(y) for y in range(2020, 2026)])

    # Step 2: Concatenate to form folder name
    folder_name = f"{level}{year}"

    # Step 3: Retrieve files from Supabase storage
    bucket_name = "markdown"

    if not supabase_available:
        st.error("Supabase connection not available. Cannot proceed.")
        return

    response = supabase.storage.from_(bucket_name).list(path=folder_name)

    # Step 4: Extract file names and let user select
    if response:
        file_names = [file['name'] for file in response if file['name'].endswith('.md')]
        if file_names:
            selected_file = st.selectbox("Select a Markdown File", file_names)

            if selected_file:
                # Full path to the file in the bucket
                file_path = f"{folder_name}/{selected_file}"

                # Download the file content
                try:
                    file_response = supabase.storage.from_(bucket_name).download(file_path)

                    if file_response:
                        markdown_content = file_response.decode("utf-8")

                        # Extract paper_no from filename, remove .md extension
                        paper_no = os.path.splitext(selected_file)[0]

                        # Process the markdown content and build UI
                        # Pass supabase client and bucket info for image fetching
                        build_ui_and_collect_metadata_revamped(
                            markdown_content,
                            supabase,
                            bucket_name,
                            folder_name,
                            paper_no
                        )

                        if st.button("Save Answers"):
                            data_for_df = []
                            if not st.session_state.input_fields_metadata:
                                st.warning("No input fields were processed to save.")

                            for field_meta in st.session_state.input_fields_metadata:
                                answer = st.session_state.get(field_meta['widget_key'], "")
                                level_extracted = field_meta['paper_no'][:2] if field_meta['paper_no'] else None

                                raw_question = field_meta['question_df']
                                cleaned_lines = []
                                for line in raw_question.splitlines():
                                    if re.match(r"^##\s*Q\d+", line.strip()):
                                        continue
                                    cleaned_line = re.sub(r'\s*#\w+', '', line)
                                    cleaned_lines.append(cleaned_line)
                                cleaned_question = "\n".join(cleaned_lines).strip()

                                data_for_df.append({
                                    "level": level_extracted,
                                    "paper": field_meta['paper_no'],
                                    "question_number": field_meta['question_number_df'],
                                    "question": cleaned_question,
                                    "answer": answer,
                                    "marks": field_meta['marks_df'],
                                    "question_type": field_meta['question_type_df']
                                })

                            if data_for_df:
                                column_order = ["level", "paper", "question_number", "question", "answer", "marks",
                                                "question_type"]
                                df = pd.DataFrame(data_for_df)
                                df = df.reindex(columns=column_order)
                                st.session_state.extracted_data = df
                                st.success("Answers extracted successfully!")
                                st.dataframe(st.session_state.extracted_data)
                            elif st.session_state.input_fields_metadata:
                                st.info("Answers saved (all were empty).")
                                st.session_state.extracted_data = pd.DataFrame(data_for_df)
                                st.dataframe(st.session_state.extracted_data)
                    else:
                        st.error("Failed to download the selected file.")

                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        else:
            st.warning("No .md files found in the selected folder.")
    else:
        st.warning("No files found in the selected folder or failed to connect to storage.")

if __name__ == "__main__":
    main() 
