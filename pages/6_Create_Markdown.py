import streamlit as st
import pymupdf
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
import os
import re # Import re module for regex operations

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize session state for storing processed markdown
if 'processed_markdown' not in st.session_state:
    st.session_state.processed_markdown = None
if 'conversion_complete' not in st.session_state:
    st.session_state.conversion_complete = False

# Convert PDF to images
def pdf_to_images(pdf_file, test_mode=False):
    images = []
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    # If test mode, only process first page
    max_pages = 1 if test_mode else doc.page_count
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        images.append((f"{i+1}", img_bytes))
    return images

# Send image to OpenAI and get markdown output
def get_markdown_from_image(image_bytes, question_label, pdf_filename):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    # Remove .pdf extension and any special characters from filename
    clean_filename = os.path.splitext(pdf_filename)[0].replace(" ", "_")
    prompt = f"""
You are helping convert scanned exam papers into structured Markdown format with lightweight placeholders for HTML/JS input field injection.

---

### 1. Image Handling
- Do **NOT** OCR or describe any diagrams or illustrations.
- Instead, insert an image reference like:
  `![Diagram of Cell X showing parts like nucleus and vacuole](media/{clean_filename}/{{question_label}}i.png)`
- Use the PDF filename '{clean_filename}' as the folder name (e.g., `media/{clean_filename}/`). The `{{question_label}}` in the image path refers to the current page number (e.g., "1", "2").
- If a single question part on a page contains multiple diagrams, label them sequentially based on order of appearance on that page: `{{question_label}}i`, `{{question_label}}ii`, etc.

---

### 2. Question Numbering (NO Type Tags)
- Carefully detect all question numbers and their sub-parts (e.g., (a), (b), (i), (ii)).
- Write each question or sub-question with just its identifier, NO type tags. Examples:
  `Q1.`
  `Q3(a).`
  `Q12(b)(i).`
- Ensure the question number (e.g., 1, 3a, 12bi) is extracted precisely.
- You must determine the question type based on clues, but do NOT append it to the question text:
  - If it has multiple-choice options (e.g., `1.`, `2.`, `A.`, `B.`) → type is `mcq`
  - If it asks for an explanation, calculation, written description → type is `oeq`
  - If it asks the student to draw, label a diagram, or plot on a graph → type is `draw`
- Never skip or omit a question number.
- Do **not** use markdown headers like `##` or `###` for questions.

---

### 3. Input Field Placeholder Insertion
- Immediately below each question or sub-question that requires a distinct answer, insert a placeholder line for the input field.
- **CRITICAL ATTRIBUTES**: All placeholders MUST use `paper_number`, `question_number`, and `question_type` attributes. Do **NOT** use the `name` attribute.
- The `paper_number` attribute in all placeholders MUST be exactly: `{clean_filename}`.
- For the `question_number` attribute in the placeholder:
    - Extract the unique identifier from the question label you have just written (e.g., "Q29(a)." or "Q5." or "Q12(b)(i).").
    - From this identifier, take ONLY the alphanumeric characters. Remove "Q", ".", "(", ")".
    - Example: If the question label is `Q29(a).`, the `question_number` attribute is "29a".
    - Example: If the question label is `Q5.`, the `question_number` attribute is "5".
    - Example: If the question label is `Q12(b)(i).`, the `question_number` attribute is "12bi".
    - The `question_number` attribute itself must NOT contain 'Q', '.', '(', ')'.
- For the `question_type` attribute, use one of: `mcq`, `oeq`, or `draw` (without the # symbol).

- Use the following lightweight placeholder formats. Pay close attention to the attributes `paper_number`, `question_number`, and `question_type`.

| Question Type | Type Value | Placeholder Format                                                                                         | Specific Instructions                                                                                                                                                                                             |
|---------------|------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Multiple Choice | `mcq`     | `[input type=text paper_number={clean_filename} question_number={{Q_PART_ALPHANUM}} question_type=mcq size=1]`             | `{{Q_PART_ALPHANUM}}` is the extracted alphanumeric question identifier (e.g., "29a", "1"). Insert only once per complete MCQ question. If an MCQ has multiple parts, place the input under the final part where the answer is expected. |
| Open-Ended      | `oeq`     | `[input type=textarea paper_number={clean_filename} question_number={{Q_PART_ALPHANUM}} question_type=oeq rows={{X}}]`         | `{{Q_PART_ALPHANUM}}` is the extracted alphanumeric question identifier (e.g., "26b"). Insert for sub-questions (e.g., `Q2(a).`) that require a direct answer. Determine `rows` (X) from 2-5 based on context, marks, or likely answer length. Default to 3 rows if unsure. |
| Drawing         | `draw`    | `[input type=canvas paper_number={clean_filename} question_number={{Q_PART_ALPHANUM}} question_type=draw rows=4]`               | `{{Q_PART_ALPHANUM}}` is the extracted alphanumeric question identifier (e.g., "27c"). Insert for sub-questions requiring a drawing. `rows` is fixed at 4.                                                      |

- **Placement Rules**: An input placeholder MUST be inserted for:
    - Every MCQ question (at the point where the single answer for it is expected).
    - Every OEQ sub-question that requires a textual answer.
    - Every DRAW sub-question that requires a drawing.
- Do **not** insert placeholders under main questions (e.g. `Q2.`) if they only serve to introduce sub-questions that will get their own placeholders.
- Never insert more than one input placeholder for the exact same question or sub-question part.

---

### 4. Text Extraction
- Transcribe only visible question text outside diagrams.
- Preserve scientific formatting, chemical equations, mathematical expressions, phrasing, and structure as accurately as possible.
- Maintain multiple-choice option formatting (e.g., `1.`, `2.`, `3.`, `4.` or `A.`, `B.`, `C.`, `D.`) and their line breaks.

---

### 5. Table Formatting
- If the question includes a table with grouped headers (e.g., "Process A/B"), simplify into a single-row format:

    ```
    |        | Process A       | Process B       | Part X        | Part Y     |
    |--------|------------------|------------------|----------------|------------|
    | (1)    | fertilization     | germination      | fruit          | seed       |
    | (2)    | fertilization     | seed dispersal   | seed           | fruit      |
    | (3)    | pollination       | seed dispersal   | seed           | fruit      |
    | (4)    | pollination       | fertilization    | fruit          | seed       |
    ```
- Do NOT attempt to merge or span cells in the markdown.
- Re-label compound headers clearly if needed for simplicity, but preserve original meaning.
- Use markdown table syntax (`|`, `---`) with aligned columns.

### 6. Output Rules
- Do NOT wrap the output in triple backticks like \`\`\`markdown.
- Do NOT use markdown headers like `##`, `###`, etc., within the question content.
- Do NOT include explanations, notes, or commentary about your process — just output clean, plain Markdown representing the exam paper content.
- Keep spacing clean and minimal — no excessive line breaks or blank lines, except where necessary for readability (e.g., between questions).
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a markdown formatting assistant for science exam papers."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=1500
    )
    return response.choices[0].message.content

# Streamlit interface
def main():
    # Main Streamlit interface
    st.title("Scanned Exam Paper → Markdown Converter")
    
    # Add test mode toggle
    test_mode = st.checkbox("Test Mode (Process First Page Only)", value=False, 
                          help="Enable this to only process the first page of the PDF for testing purposes")

    uploaded_pdf = st.file_uploader("Upload scanned PDF", type=["pdf"])

    # Reset conversion state if new file is uploaded
    if uploaded_pdf and st.session_state.processed_markdown is not None:
        st.session_state.processed_markdown = None
        st.session_state.conversion_complete = False

    # Only show conversion button if file is uploaded and not yet processed
    if uploaded_pdf and not st.session_state.conversion_complete:
        if st.button("Start Conversion"):
            # Define clean_filename for use in post-processing
            clean_filename = os.path.splitext(uploaded_pdf.name)[0].replace(" ", "_")
            
            with st.spinner("Converting PDF pages to images..." + (" (Test Mode: First Page Only)" if test_mode else "")):
                images = pdf_to_images(uploaded_pdf, test_mode)

                all_markdown_parts = []
                for idx, (label, img_bytes) in enumerate(images):
                    with st.spinner(f"Processing page {label} of {len(images)}..."):
                        md = get_markdown_from_image(img_bytes, label, uploaded_pdf.name)
                        all_markdown_parts.append(md)

                # Join all markdown parts initially
                raw_markdown = "\n\n".join(all_markdown_parts)
                
                # Post-process to add headers for main questions and fix image filenames
                processed_lines = []
                current_question_number = None
                
                for line in raw_markdown.splitlines():
                    # Track current question number for image filename correction
                    question_match = re.match(r"^Q(\d+)(?:\([a-zA-Z0-9]+\))?\.\s*", line)
                    if question_match:
                        # Extract just the main question number (e.g., "5" from "Q5(a).")
                        current_question_number = question_match.group(1)
                    
                    # Check if this is a main question line (without sub-parts) to add header
                    main_question_match = re.match(r"^(Q\d+)\.\s*", line)
                    if main_question_match and not re.search(r"Q\d+\([a-zA-Z0-9]+\)", line):
                        question_number_part = main_question_match.group(1) # e.g. "Q1"
                        processed_lines.append(f"## {question_number_part}")
                    
                    # Fix image filenames to use current question number instead of page number
                    if current_question_number and "![" in line and f"media/{clean_filename}/" in line:
                        # Replace page number with question number in image paths
                        # Pattern: media/{clean_filename}/PAGENUMBERi.png -> media/{clean_filename}/QUESTIONNUMBERi.png
                        line = re.sub(
                            rf"(media/{re.escape(clean_filename)}/)\d+([a-z]*\.png)",
                            rf"\g<1>{current_question_number}\g<2>",
                            line
                        )
                    
                    processed_lines.append(line)
                
                st.session_state.processed_markdown = "\n".join(processed_lines)
                st.session_state.conversion_complete = True
                st.success("Conversion complete!" + (" (Test Mode)" if test_mode else ""))

    # Display results if conversion is complete
    if st.session_state.conversion_complete and st.session_state.processed_markdown:
        st.markdown("### Markdown Output")
        st.markdown(st.session_state.processed_markdown)
        
        # Add download button for the markdown file
        st.download_button(
            label="Download Markdown File",
            data=st.session_state.processed_markdown,
            file_name=f"{os.path.splitext(uploaded_pdf.name)[0]}.md",
            mime="text"
        )

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    main()
else:
    st.warning("Please log in first before using OO AI.") 