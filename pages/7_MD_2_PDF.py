import streamlit as st
import os
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from markdown import markdown
import tempfile
import shutil
from bs4 import BeautifulSoup
import io
import urllib.parse
from reportlab.pdfbase.pdfmetrics import stringWidth

def extract_questions_from_markdown(md_text):
    """Extract questions from markdown text, handling headers and content properly."""
    # Split the text by question headers
    # This pattern matches "## Q" followed by numbers and optional subparts
    parts = re.split(r'(##\s*Q\d+(?:\([a-z]\))?\.?)', md_text)
    
    questions = []
    current_question = None
    current_content = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Check if this part is a question header
        if re.match(r'^##\s*Q\d+(?:\([a-z]\))?\.?$', part):
            # If we have a previous question, save it
            if current_question and current_content:
                questions.append({
                    "label": current_question,
                    "content": "\n".join(current_content).strip()
                })
            # Start new question
            current_question = part.replace("##", "").strip()
            current_content = []
        else:
            # This is content for the current question
            if current_question is not None:
                current_content.append(part)
    
    # Don't forget the last question
    if current_question and current_content:
        questions.append({
            "label": current_question,
            "content": "\n".join(current_content).strip()
        })
    
    return questions

# Register Arial font
def register_arial_font():
    try:
        # Try to register Arial font from Windows system
        arial_path = os.path.join(os.environ['WINDIR'], 'Fonts', 'arial.ttf')
        arial_bold_path = os.path.join(os.environ['WINDIR'], 'Fonts', 'arialbd.ttf')
        pdfmetrics.registerFont(TTFont('Arial', arial_path))
        pdfmetrics.registerFont(TTFont('Arial-Bold', arial_bold_path))
        return True
    except:
        # Fallback to Helvetica if Arial is not available
        return False

def get_available_width(doc):
    """Calculate available width for content considering margins."""
    return doc.width - (doc.leftMargin + doc.rightMargin)

def calculate_text_width(text, font_name, font_size):
    """Calculate the width of text in points."""
    return stringWidth(text, font_name, font_size)

def get_wrapped_text_width(text, font_name, font_size, max_width):
    """Calculate the width needed for wrapped text."""
    words = text.split()
    if not words:
        return 0
    
    # Start with the first word
    current_line = words[0]
    max_line_width = calculate_text_width(current_line, font_name, font_size)
    
    for word in words[1:]:
        # Try adding the next word
        test_line = current_line + " " + word
        test_width = calculate_text_width(test_line, font_name, font_size)
        
        if test_width <= max_width:
            # Word fits on current line
            current_line = test_line
            max_line_width = max(max_line_width, test_width)
        else:
            # Word needs to go on next line
            max_line_width = max(max_line_width, calculate_text_width(current_line, font_name, font_size))
            current_line = word
    
    return max_line_width

def split_long_text(text, max_chars_per_line=30):
    """Split long text into multiple lines at word boundaries."""
    # Replace special characters with spaces to help with splitting
    text = text.replace('➝', ' ➝ ')  # Add spaces around arrows
    
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        # If adding this word would exceed the limit, start a new line
        if current_length + len(word) + 1 > max_chars_per_line and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += len(word) + 1 if current_line else len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Clean up the text by removing extra spaces around arrows
    result = '\n'.join(lines)
    result = result.replace(' ➝ ', '➝')  # Remove spaces around arrows
    return result

def calculate_column_widths(data, font_name, font_size, available_width, min_col_width=40, max_col_width=150):
    """Calculate optimal column widths based on content."""
    if not data or not data[0]:
        return []
    
    num_cols = len(data[0])
    col_widths = [0] * num_cols
    
    # First pass: calculate minimum width needed for each column
    for row_idx, row in enumerate(data):
        for col_idx, cell_text in enumerate(row):
            # Split cell text into lines and calculate width for each line
            lines = cell_text.split('\n')
            max_line_width = 0
            
            # For header row, use a shorter max width to encourage wrapping
            max_chars = 30 if row_idx == 0 else 100
            
            for line in lines:
                # Handle special characters in width calculation
                line = line.replace('➝', '→')  # Use a standard arrow for width calculation
                line_width = calculate_text_width(line.strip(), font_name, font_size)
                max_line_width = max(max_line_width, line_width)
            
            # Add padding
            max_line_width += 16  # Increased padding
            col_widths[col_idx] = max(col_widths[col_idx], max_line_width)
    
    # Special handling for the first column (row labels)
    if col_widths:
        col_widths[0] = max(col_widths[0], 40)  # Ensure first column is wide enough for labels
    
    # Ensure minimum and maximum widths
    col_widths = [max(min_col_width, min(width, max_col_width)) for width in col_widths]
    
    # If total width exceeds available width, scale down proportionally
    total_width = sum(col_widths)
    if total_width > available_width:
        scale_factor = available_width / total_width
        col_widths = [width * scale_factor for width in col_widths]
    
    return col_widths

def convert_markdown_to_reportlab_elements(markdown_text, styles, doc, base_path=None, font_size=11, font_name='Arial', font_name_bold='Arial-Bold', line_spacing=13):
    """Convert markdown text to ReportLab elements."""
    # Convert markdown to HTML, preserving line breaks and handling special characters
    html = markdown(markdown_text, extensions=['tables', 'fenced_code', 'nl2br'])
    soup = BeautifulSoup(html, 'html.parser')
    
    elements = []
    available_width = get_available_width(doc)
    
    # Process each element
    for element in soup.find_all(['h1', 'h2', 'p', 'table', 'ul', 'ol', 'img']):
        if element.name == 'h1':
            elements.append(Paragraph(element.get_text(), styles['CustomHeading1']))
            elements.append(Spacer(1, 0.5*cm))
        elif element.name == 'h2':
            elements.append(Paragraph(element.get_text(), styles['CustomHeading1']))
            elements.append(Spacer(1, 0.3*cm))
        elif element.name == 'p':
            # Handle paragraphs with proper line breaks
            text = element.get_text(separator='\n')
            # Split by newlines and process each line
            lines = text.split('\n')
            processed_lines = []
            
            for line in lines:
                line = line.strip()
                if line:
                    # Check if line starts with a letter followed by colon (e.g., "A:", "B:")
                    if re.match(r'^[A-Z]:', line):
                        # Add extra space after the colon
                        line = line.replace(':', ': ', 1)
                    processed_lines.append(line)
            
            # Join lines with proper spacing
            text = '\n'.join(processed_lines)
            elements.append(Paragraph(text, styles['CustomNormal']))
            elements.append(Spacer(1, 0.2*cm))
        elif element.name == 'table':
            # Convert table to ReportLab table
            data = []
            header_row = None
            
            # First, process the header row to split long text
            for row_idx, row in enumerate(element.find_all('tr')):
                cells = []
                for cell in row.find_all(['td', 'th']):
                    # Get text content, preserving special characters
                    cell_text = cell.get_text(separator='\n')
                    # Clean up whitespace while preserving line breaks
                    cell_text = '\n'.join(line.strip() for line in cell_text.split('\n'))
                    
                    # Split long header text into multiple lines
                    if row_idx == 0:  # Header row
                        # Split headers into more lines for better readability
                        cell_text = split_long_text(cell_text, max_chars_per_line=30)
                    
                    cells.append(cell_text)
                
                if row_idx == 0:
                    header_row = cells
                else:
                    data.append(cells)
            
            if header_row:
                data.insert(0, header_row)
            
            if data:
                # Calculate column widths with adjusted parameters
                col_widths = calculate_column_widths(
                    data,
                    font_name,
                    font_size,
                    get_available_width(doc),
                    min_col_width=40,  # Increased minimum width
                    max_col_width=150  # Adjusted maximum width
                )
                
                # Create table with calculated widths
                table = Table(data, colWidths=col_widths, repeatRows=1)
                
                # Create table style with improved formatting
                table_style = [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), font_name_bold),
                    ('FONTSIZE', (0, 0), (-1, -1), font_size),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                    ('FONTNAME', (0, 1), (-1, -1), font_name),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('PADDING', (0, 0), (-1, -1), 8),  # Increased padding
                    ('WORDWRAP', (0, 0), (-1, -1), True),
                    ('LEFTPADDING', (0, 0), (-1, -1), 8),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ('LEADING', (0, 0), (-1, -1), line_spacing),
                ]
                
                # Add specific styles for header row
                table_style.extend([
                    ('FONTNAME', (0, 0), (-1, 0), font_name_bold),
                    ('TOPPADDING', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('VALIGN', (0, 0), (-1, 0), 'MIDDLE'),
                ])
                
                # Add specific style for the first column (row labels)
                table_style.extend([
                    ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (0, -1), font_name_bold),
                ])
                
                table.setStyle(TableStyle(table_style))
                elements.append(table)
                elements.append(Spacer(1, 0.3*cm))
        elif element.name in ['ul', 'ol']:
            for li in element.find_all('li'):
                bullet = '• ' if element.name == 'ul' else f'{element.find_all("li").index(li) + 1}. '
                elements.append(Paragraph(bullet + li.get_text(), styles['CustomNormal']))
            elements.append(Spacer(1, 0.2*cm))
        elif element.name == 'img':
            # Handle images from markdown paths
            src = element.get('src', '')
            if src:
                try:
                    # Handle both relative and absolute paths
                    if base_path and not os.path.isabs(src):
                        img_path = os.path.join(base_path, src)
                    else:
                        img_path = src
                    
                    # URL decode the path if needed
                    img_path = urllib.parse.unquote(img_path)
                    
                    if os.path.exists(img_path):
                        # Calculate image size to fit page width
                        img = Image(img_path)
                        img_width = img.imageWidth
                        img_height = img.imageHeight
                        
                        # Scale image if it's too wide
                        if img_width > available_width:
                            scale = available_width / img_width
                            img_width *= scale
                            img_height *= scale
                        
                        img = Image(img_path, width=img_width, height=img_height)
                        elements.append(img)
                        elements.append(Spacer(1, 0.3*cm))
                except Exception as e:
                    st.warning(f"Could not load image: {src}. Error: {str(e)}")
    
    return elements

def create_questions_pdf(questions, output_path, base_path=None):
    """Create a single PDF with one question per page using ReportLab."""
    # Register Arial font
    has_arial = register_arial_font()
    font_name = 'Arial' if has_arial else 'Helvetica'
    font_name_bold = 'Arial-Bold' if has_arial else 'Helvetica-Bold'
    
    # Define consistent font sizes
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_HEADING = 11
    LINE_SPACING = 13
    
    # Create custom styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomHeading1',
        parent=styles['Heading1'],
        fontName=font_name_bold,
        fontSize=FONT_SIZE_HEADING,
        spaceAfter=30,
        textColor=colors.black,
        leading=LINE_SPACING
    ))
    styles.add(ParagraphStyle(
        name='CustomNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=FONT_SIZE_NORMAL,
        leading=LINE_SPACING,
        spaceAfter=12
    ))
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Build the document content
    story = []
    
    for question in questions:
        # Add question content
        elements = convert_markdown_to_reportlab_elements(
            f"# {question['label']}\n\n{question['content']}",
            styles,
            doc,
            base_path,
            font_size=FONT_SIZE_NORMAL,  # Pass font size to conversion function
            font_name=font_name,
            font_name_bold=font_name_bold,
            line_spacing=LINE_SPACING
        )
        story.extend(elements)
        
        # Add page break after each question except the last one
        if question != questions[-1]:
            story.append(PageBreak())
    
    # Build the PDF
    doc.build(story)

def main():
    st.title("Markdown to PDF Converter")
    
    uploaded_md = st.file_uploader("Upload Markdown (.md)", type=["md"])
    
    if uploaded_md:
        # Create temporary directory
        tmp_dir = Path("tmp_pdf_output")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
        
        # Get the base path for images (same directory as the markdown file)
        base_path = None
        if hasattr(uploaded_md, 'name'):
            base_path = os.path.dirname(os.path.abspath(uploaded_md.name))
        
        # Read and process markdown
        md_text = uploaded_md.read().decode("utf-8")
        questions = extract_questions_from_markdown(md_text)
        
        if questions:
            st.success(f"Found {len(questions)} questions")
            
            # Create single PDF with all questions
            output_path = tmp_dir / "questions.pdf"
            create_questions_pdf(questions, str(output_path), base_path)
            
            # Display preview of questions
            for i, question in enumerate(questions, 1):
                st.markdown(f"**{question['label']}**")
                st.markdown(question['content'][:200] + "..." if len(question['content']) > 200 else question['content'])
                st.markdown("---")
            
            # Add download button for the complete PDF
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Complete PDF",
                    f,
                    file_name="questions.pdf",
                    mime="application/pdf"
                )
        else:
            st.error("No questions found in the markdown file")

if __name__ == "__main__":
    main()
