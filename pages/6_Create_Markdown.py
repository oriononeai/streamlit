import streamlit as st
import pymupdf
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
import os

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
        images.append((f"Q{i+1}", img_bytes))
    return images

# Send image to OpenAI and get markdown output
def get_markdown_from_image(image_bytes, question_label, pdf_filename):
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    # Remove .pdf extension and any special characters from filename
    clean_filename = os.path.splitext(pdf_filename)[0].replace(" ", "_")
    prompt = f"""
You are helping convert scanned exam papers into Markdown format.

### 1. Image Handling
- Do NOT OCR or describe any diagrams or illustrations.
- Instead, insert an actual image reference like:
`![Diagram of Cell X showing parts like nucleus and vacuole](media/{clean_filename}/{{question_label}}i.png)`
- Use the PDF filename '{clean_filename}' as the folder name (e.g. media/{clean_filename}/).
- If the question contains multiple diagrams, label them as `{{question_label}}i`, `{{question_label}}ii`, etc., based on order of appearance.

### 2. Question Numbering
- Carefully detect all question numbers, even if they are not explicitly labeled as "Q".
- Prefix each question with a clear label in this format: `Q1.`, `Q2.`, `Q3.`, etc.
- For sub-parts like (a), (b), format them as: `Q3(a).`, `Q3(b).`, etc.
- Do NOT use markdown headers like `## Q1`. Just use plain inline text (e.g., `Q1.`, not `## Q1`).
- If a question number is visible (e.g., "39.") at the top of the question or near the image, use it (e.g., `Q39.`).
- Never skip a question number. Always include every question and sub-part.

### 3. Text Extraction
- Only transcribe the question text *outside* the image or diagram.
- Preserve all scientific formatting and phrasing as much as possible.
- Maintain multiple-choice answer formatting (e.g., `1.`, `2.`, `3.`, `4.`).

### 4. Table Formatting
- If the question includes a table with grouped headers (e.g., "Process" over A/B and "Part" over X/Y), simplify the header into a flat single row:
    ```
    |        | Process A       | Process B       | Part X        | Part Y     |
    |--------|------------------|------------------|----------------|------------|
    | (1)    | fertilization     | germination      | fruit          | seed       |
    | (2)    | fertilization     | seed dispersal   | seed           | fruit      |
    | (3)    | pollination       | seed dispersal   | seed           | fruit      |
    | (4)    | pollination       | fertilization    | fruit          | seed       |
    ```
- Do NOT attempt to merge or span cells.
- Re-label compound headers clearly.
- Use markdown table syntax (`|`, `---`) with aligned columns.

### 5. Output Rules
- Do NOT wrap the output in triple backticks like ```markdown.
- Do NOT use markdown headers like `##`, `###`, etc.
- Do NOT include explanations, notes, or commentary — just output clean, plain Markdown.
- Keep spacing clean and minimal — no excessive line breaks or blank lines.
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
            with st.spinner("Converting PDF pages to images..." + (" (Test Mode: First Page Only)" if test_mode else "")):
                images = pdf_to_images(uploaded_pdf, test_mode)

                all_markdown = ""
                for idx, (label, img_bytes) in enumerate(images):
                    with st.spinner(f"Processing {label}..."):
                        md = get_markdown_from_image(img_bytes, label, uploaded_pdf.name)
                        all_markdown += f"\n\n## {label}\n{md}"

                # Store the processed markdown in session state
                st.session_state.processed_markdown = all_markdown
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
            file_name="exam_paper.md",
            mime="text"
        )

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    main()
else:
    st.warning("Please log in first before using OO AI.") 