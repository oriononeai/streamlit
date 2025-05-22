import streamlit as st
import pandas as pd
import json
import base64
from openai import OpenAI
from io import BytesIO
import os
import re

# -------------------------------------------------------------------
# IMPORTANT: You must have access to GPT-4 Vision (not publicly available).
# This code will not run successfully on standard OpenAI plans.
# -------------------------------------------------------------------

# OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

def process_image_with_openai(image_file):
    """
    Returns a Python dictionary with the structure:
    {
        "questions": [
            {"question_number": "1", "answer": "4"},
            {"question_number": "2", "answer": "2"},
            ...
        ]
    }
    """
    #imgbuffered = BytesIO()
    #image_file.save(imgbuffered,format="PNG")  # Save image to BytesIO buffer in JPEG format (or any format you need)
    #image_bytes = imgbuffered.getvalue()  # Get the bytes of the image

    # Read the image bytes
    image_bytes = image_file.read()

    # Optionally, you can reset the file pointer if needed:
    image_file.seek(0)

    # Convert image bytes to base64
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    # Construct a prompt for GPT-4 Vision
    # This prompt instructs the model to decode the base64 image, read the table,
    # and return the question numbers and answers in JSON format.
    # The exact prompt wording and structure may require tuning for best results.
    prompt = f"""
            The user has provided this image in base64 format.
            Please decode this image, read the table of question numbers and answers, and return them in valid JSON format with the structure:
            {{
              "questions": [
                {{
                  "question_number": question number,
                  "answer": answer
                }},
                ...
              ]
            }}.
            
            Only include question/answer pairs you are confident about. If there is a "Q" with the question number, remove the "Q" and just return the number as the question number.
            If there is any ambiguity, do not give any answer. Do not include any additional commentary.
                """

    try:
        client = OpenAI()

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
        )

        reply = completion.choices[0].message.content.strip()
        # Define the marker to search for
        pattern = re.compile(r'\}\s*\]\s*\}')

        # Extract the classification answer.
        response = reply[reply.find('{'):]  # remove all char before {
        #response = response.split('}', 1)[0] + '}'  # remove all char after }
        match = pattern.search(response)
        if match:
            # Use the end position of the match to slice the string
            response_clean = response[:match.end()]
        else:
            response_clean = response
        try:
            json_data = json.loads(response_clean)
        except Exception as e:
            st.error(f"error as {e}")

        return json_data
    except Exception as e:
        st.error(f"Error calling GPT-4o: {e}")
        return {"questions": []}

def main():
    st.title("SM AI-Tutor (Hypothetical GPT-4 Vision)")

    # Step 1: User Inputs
    level = st.selectbox("Select Primary Level", ["P3", "P4", "P5", "P6"])
    year = st.text_input("Year")
    school = st.selectbox("Select School", [
        "ACSP", "ATPS", "HPPS", "MBPS", "MGSP", "NHPS", "ACSJ",
        "NYPS", "RGPS", "SCGS", "SHPS", "SJIJ", "TNPS", "RSPS",
        "CHSP", "PCPS", "RSSP"
    ])
    paper_type = st.text_input("Type of Paper, WA2, SA1 etc")

    paper_code = f"{level}{year}{school}{paper_type}"

    st.write(paper_code)

    # Step 2: Concatenate inputs to create the 'paper' code
    if st.button("Generate Paper Code"):
        if year and paper_type:
            paper_code = f"{level}{year}{school}{paper_type}"
            st.session_state.paper_code = paper_code  # Store for later use
            st.success(f"Generated Paper Code: {paper_code}")
        else:
            st.warning("Please provide both Year and Type of Paper to generate the code.")

    # Step 3: Upload screenshot image of the answer sheet
    image_file = st.file_uploader("Upload Screenshot Image of Answer Sheet", type=["png", "jpg", "jpeg"])
    if image_file:
        st.image(image_file, caption="Uploaded Answer Sheet", use_container_width=True)

    # Step 4 & 5: Process the image using GPT-4 with vision
    if st.button("Process Answer Sheet"):
        if not image_file:
            st.warning("Please upload an image file first.")
        elif "paper_code" not in st.session_state:
            st.warning("Please generate the paper code first.")
        else:
            result = process_image_with_openai(image_file)
            st.write("OpenAI Response (JSON):")
            #st.json(result)

            # Step 6: Tabulate the results into a table
            data = []
            for question in result.get("questions", []):
                row = {
                    "paper": st.session_state.paper_code,
                    "question_number": question.get("question_number", ""),
                    "level": level,
                    "question": "",  # Placeholder for the question text
                    "answer": question.get("answer", ""),
                    "marks": 2,
                    "question_type": "multiple_choice"
                }
                data.append(row)
            df = pd.DataFrame(data)
            st.session_state.extracted_data = df
            st.write("Extracted Data:")
            st.dataframe(df)

# Authentication check (replace with your actual authentication logic)
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False  # Replace with your authentication logic

if st.session_state.authenticated:
    main()
else:
    st.warning("Please log in first before using SM AI-Tutor.")
