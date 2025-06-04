import streamlit as st
# import pymupdf # Removed
from openai import OpenAI
import os
import json
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
import time

# --- Environment Selection (Copied from 4_Structure_DB_OEQ.py) ---
ENV_OPTIONS = ["QA", "PROD"]
# Place selectbox in sidebar
selected_env = st.sidebar.selectbox("Select Environment", ENV_OPTIONS, index=0, key="create_answer_env_select")

# --- Constants (Copied and adapted from 4_Structure_DB_OEQ.py) ---
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

# --- Initialize Supabase Client (Copied from 4_Structure_DB_OEQ.py) ---
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

# --- Initialize OpenAI Client (Copied from 4_Structure_DB_OEQ.py) ---
openai_client: OpenAI | None = None
openai_available = False
if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        openai_available = True
    except Exception as e:
        st.sidebar.error(f"Failed to initialize OpenAI client: {e}")
else:
    st.sidebar.warning("OPENAI_API_KEY missing. Cannot generate model answers.")

# Initialize session state variables
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = pd.DataFrame(columns=[
        'level', 'paper', 'question_number', 'question', 
        'answer', 'marks', 'question_type', 'astructure'
    ])

if 'review_data' not in st.session_state:
    st.session_state.review_data = pd.DataFrame(columns=[
        'level', 'paper', 'question_number', 'question', 'answer', 
        'marks', 'question_type', 'structured_answer',
        'evaluation_score', 'evaluation_status', 'evaluation_issues',
        'evaluation_suggested_fix', 'review_status',
        'revision_count', 'last_revision', 'revision_error'
    ])

# --- Helper Functions (Copied and adapted from 4_Structure_DB_OEQ.py) ---

def get_embedding(text):
    response = openai_client.embeddings.create(model = "text-embedding-3-small", input=text)
    return response.data[0].embedding

def retrieve_similar_cases(level, new_q_embedding):
    """
    Retrieve similar cases using vector similarity search with the vsearch column.
    
    For optimal performance, create this function in your Supabase database:
    
    CREATE OR REPLACE FUNCTION match_similar_questions(
      query_embedding vector(1536),
      query_level text,
      match_count int DEFAULT 3
    )
    RETURNS TABLE (
      question text,
      answer text,
      astructure text,
      similarity float
    )
    LANGUAGE sql
    AS $$
      SELECT 
        question,
        answer,
        astructure,
        1 - (vsearch <=> query_embedding) as similarity
      FROM pri_sci_paper
      WHERE level = query_level
        AND answer IS NOT NULL
        AND astructure IS NOT NULL
        AND vsearch IS NOT NULL
      ORDER BY vsearch <=> query_embedding
      LIMIT match_count;
    $$;
    """
    try:
        # Method 1: Try using RPC call (requires database function)
        try:
            response = supabase.rpc('match_similar_questions', {
                'query_embedding': new_q_embedding,
                'query_level': level,
                'match_count': 3
            }).execute()
            
            if response.data:
                return response.data
        except Exception as rpc_error:
            st.info(f"RPC method not available: {rpc_error}")
        
        # Method 2: Try using PostgREST vector similarity (if configured)
        try:
            # Format embedding as PostgreSQL array
            embedding_str = '[' + ','.join(map(str, new_q_embedding)) + ']'
            
            # Use PostgREST with vector similarity
            response = supabase.table(TARGET_TABLE) \
                              .select("question, answer, astructure") \
                              .eq("level", level) \
                              .not_.is_("answer", "null") \
                              .not_.is_("astructure", "null") \
                              .not_.is_("vsearch", "null") \
                              .order(f"vsearch <-> '{embedding_str}'::vector") \
                              .limit(3) \
                              .execute()
            
            if response.data:
                return response.data
        except Exception as vector_error:
            st.info(f"Vector similarity query failed: {vector_error}")
        
        # Method 3: Fallback to basic filtering by level
        st.info(f"Using fallback method for level {level}")
        response = supabase.table(TARGET_TABLE) \
                          .select("question, answer, astructure") \
                          .eq("level", level) \
                          .not_.is_("answer", "null") \
                          .not_.is_("astructure", "null") \
                          .not_.is_("vsearch", "null") \
                          .limit(10) \
                          .execute()
        
        # Return first 3 as fallback
        return response.data[:3] if response.data else []
        
    except Exception as e:
        st.warning(f"Error in retrieve_similar_cases for level {level}: {e}")
        return []

def fetch_non_mcq_questions(supabase_client: Client, paper_code: str) -> list[dict]:
    """Fetches non-MCQ questions for a specific paper."""
    if not supabase_available or not supabase_client:
        st.error("Supabase connection not available.")
        return []
    try:
        # Select necessary fields, filter by paper and exclude MCQ/multiple_choice
        response = supabase_client.table(TARGET_TABLE) \
                                  .select("*") \
                                  .eq("paper", paper_code) \
                                  .not_.in_("question_type", ["MCQ", "multiple_choice"]) \
                                  .order("question_number", desc=False) \
                                  .execute()
        if response.data:
            # Filter out rows with missing essential data needed for answer generation
            valid_rows = [
                row for row in response.data 
                if row.get('question') # Ensure question exists
            ]
            count_total = len(response.data)
            count_valid = len(valid_rows)
            if count_total > count_valid:
                 st.warning(f"Fetched {count_total} non-MCQ rows, but {count_total - count_valid} rows are missing 'question' text and will be skipped.")
            return valid_rows
        else:
            st.info(f"No non-MCQ questions found for paper '{paper_code}' or an error occurred.")
            return []
    except Exception as e:
        st.error(f"Error fetching questions for paper {paper_code}: {e}")
        return []

def generate_answers_batch_with_openai(questions_list: list) -> dict:
    """Generate model answers for all questions at once using OpenAI with similar cases as examples."""
    if not openai_available or not openai_client:
        return {"error": "OpenAI client not available"}
    
    try:
        # Same system prompt logic as individual processing
        system_prompt = """You are a Primary School Science expert teacher. Generate concise and accurate model answers and writing structured model answers based on curriculum expectations.
            IMPORTANT: Consider the context between related questions (e.g., 29a, 29b, 29c) as they may build upon each other.

            ## Structured Answer Generation Guidelines for Primary Science Questions

                ### 1. Question Type Classification
                First, classify each question into one of these 4 types:
                
                **A. "fact_recall"** - Basic recalling of scientific facts
                - Simple identification, naming, or listing questions
                - Example: "What is shown in the picture?", "Name three types of energy"
                - Specify number of answers using number_required field
                
                **B. "explanation"** - Questions requiring scientific reasoning
                - Require explanation of cause-effect relationships, processes, or phenomena  
                - Must include relevant fields: decision, cause, effect, object
                - Example: "Explain why plants need sunlight", "Why does ice melt?"
                
                **C. "experimental_purpose"** - Experimental design and scientific method questions
                - Questions about experiments, investigations, or scientific procedures
                - Must include: decision, goal, independent_variable, dependent_variable, controlled_variable
                - Example: "Design an experiment to test...", "What variables should be controlled?"
                
                **D. "range"** - Questions where answer falls within a numerical range
                - Answers that involve measurements, quantities, or ranges (x to y)
                - Example: "The temperature ranges from __ to __", "How many grams..."

                ### 2. Structure of Model Answers by Question Type
                
                **For "fact_recall" questions:**
                - Focus on the "answer" field with clear, factual information
                - Use "number_required" to specify how many answers are needed
                - Other fields can be left empty if not applicable
                
                **For "explanation" questions:**
                - Include relevant fields: decision, cause, effect, object
                - Ensure scientific completeness in cause-effect relationships
                - All fields must have sufficient scientific detail
                
                **For "experimental_purpose" questions:**
                - Must include: decision, goal, independent_variable, dependent_variable, controlled_variable  
                - Missing or inaccurate fields result in incomplete answers
                - Each field requires sufficient scientific completeness
                
                **For "range" questions:**
                - Focus on providing the appropriate range or numerical answer
                - Include units and context where relevant

                Leave any field empty (`[]` or `""`) if it does not apply to the specific question type.

                ---

                ### 3. Scientific Accuracy and Completeness
                - All scientific concepts must be clearly stated, with correct terminology and logical cause-effect relationships.
                - Use precise scientific terms. Avoid vague or incorrect substitutes.
                - E.g., say "evaporation" instead of "disappears," or "diffusion" instead of "spreads."
                - Ensure the explanation includes:
                1. The event (what is happening),
                2. The cause (why it happens),
                3. The result (what it leads to),
                4. The related object (what is involved or affected).

                ---

                ### 4. Critical Scientific Accuracy Rules

                #### Scientific Transport Processes
                - When describing processes like oxygen or nutrients being delivered to cells, mention the correct carrier (e.g., **blood**).
                - Correct: "The heart pumps blood containing oxygen to the cells."
                - Incorrect: "The heart pumps oxygen to the cells."

                #### Precision of Scientific Terms
                - Specific scientific relationships must be expressed with accurate verbs and descriptions.
                - E.g., "Water evaporates as it gains heat" is better than "Water goes away."

                #### Biological Context Accuracy
                - Use scientifically appropriate context.
                - E.g., For pollination, "same species" or "same type of plant" is correct; "same plant" is biologically inaccurate.

                ---

                ### 5. General Answering Guidelines
                - Answers should be **clear, concise, and complete**.
                - Avoid unnecessary repetition.
                - Do not include personal opinions, assumptions, or praise.

                ---

    
                ### 6. Return your response as a JSON object with this format:
                {
                    "questions_answers": {
                        "question_number_1": {
                            "question_type": "explanation",
                            "model_answer": "Your concise, scientifically accurate answer here",
                            "structured_answer": {
                                "decision": [],
                                "cause": ["The sun provides heat"],
                                "effect": ["Water particles gain energy and evaporate"],
                                "object": [],
                                "goal": "",
                                "independent_variable": "",
                                "dependent_variable": "",
                                "controlled_variable": "",
                                "answer": ["Water evaporates because the sun provides heat, causing the water particles to gain energy and change into water vapour."],
                                "number_required": 1
                            }
                        },
                        "question_number_2": {
                            "question_type": "fact_recall",
                            "model_answer": "Your concise, scientifically accurate answer here",
                            "structured_answer": {
                                "decision": [],
                                "cause": [],
                                "effect": [],
                                "object": [],
                                "goal": "",
                                "independent_variable": "",
                                "dependent_variable": "",
                                "controlled_variable": "",
                                "answer": ["List item 1", "List item 2", "List item 3"],
                                "number_required": 3
                            }
                        },
                        "question_number_3": {
                            "question_type": "experimental_purpose", 
                            "model_answer": "Your concise, scientifically accurate answer here",
                            "structured_answer": {
                                "decision": ["Use a fair test method"],
                                "cause": [],
                                "effect": [],
                                "object": [],
                                "goal": "To determine which material dissolves fastest",
                                "independent_variable": "Type of material being tested",
                                "dependent_variable": "Time taken to dissolve completely",
                                "controlled_variable": "Amount of water, temperature, stirring speed",
                                "answer": ["To find out which material dissolves fastest, use equal amounts of each material in the same volume of water at the same temperature, stirring at the same speed, and measure the time taken for complete dissolution."],
                                "number_required": 1
                            }
                        },
                        "question_number_4": {
                            "question_type": "range",
                            "model_answer": "Your concise, scientifically accurate answer here", 
                            "structured_answer": {
                                "decision": [],
                                "cause": [],
                                "effect": [],
                                "object": [],
                                "goal": "",
                                "independent_variable": "",
                                "dependent_variable": "",
                                "controlled_variable": "",
                                "answer": ["The normal human body temperature ranges from 36.1°C to 37.2°C"],
                                "number_required": 1
                            }
                        }
                        // ... continue for all questions
                    }
                }
                
                Now generate a model answer based on the question provided, following the above structure and question type classification."""

        # Step 1: Get existing embeddings from vsearch column instead of generating new ones
        st.info("🔄 Fetching existing embeddings from database...")
        questions_with_examples = []
        skipped_questions = []
        
        # Check embedding availability first
        questions_with_embeddings = 0
        questions_without_embeddings = 0
        
        for question_data in questions_list:
            if question_data.get('vsearch'):
                questions_with_embeddings += 1
            else:
                questions_without_embeddings += 1
        
        st.info(f"📊 Embedding availability: {questions_with_embeddings} questions have embeddings, {questions_without_embeddings} questions missing embeddings")
        
        if questions_without_embeddings > 0:
            st.warning(f"⚠️ {questions_without_embeddings} questions are missing embeddings in the vsearch column. Consider running the Vector Paper script (9_Vector_Paper.py) to generate embeddings first.")
        
        for i, question_data in enumerate(questions_list):
            question_text = question_data.get('question', '')
            question_number = question_data.get('question_number', 'Unknown')
            level = question_data.get('level', '')
            paper = question_data.get('paper', '')
            
            # Get the existing embedding from vsearch column
            existing_embedding = question_data.get('vsearch', None)
            
            if not existing_embedding or not level:
                skipped_questions.append({
                    'question_number': question_number,
                    'reason': 'Missing vsearch embedding or level in database'
                })
                # Still include the question but without examples
                questions_with_examples.append({
                    'question_data': question_data,
                    'similar_cases': []
                })
                continue
            
            # Get similar cases for this question using its existing embedding
            try:
                similar_cases = retrieve_similar_cases(level, existing_embedding)
                questions_with_examples.append({
                    'question_data': question_data,
                    'similar_cases': similar_cases
                })
            except Exception as e:
                st.warning(f"Failed to retrieve similar cases for question {question_number}: {e}")
                skipped_questions.append({
                    'question_number': question_number,
                    'reason': f'Similar cases retrieval failed: {e}'
                })
                questions_with_examples.append({
                    'question_data': question_data,
                    'similar_cases': []
                })
        
        # Step 2: Build user content with questions and their similar cases
        questions_text = "Here are all the Primary School Science questions from this paper:\n\n"
        
        for i, item in enumerate(questions_with_examples, 1):
            question_data = item['question_data']
            similar_cases = item['similar_cases']
            
            question_text = question_data.get('question', 'N/A')
            question_number = question_data.get('question_number', 'Unknown')
            question_type = question_data.get('question_type', 'Unknown')
            marks = question_data.get('marks', 'Not specified')
            
            questions_text += f"""Question {i}:
            Question Number: {question_number}
            Question Type: {question_type}
            Marks: {marks}
            Question: {question_text}
            """
            
            # Add similar cases as examples if available
            if similar_cases:
                examples_text = format_similar_cases_as_json_examples(similar_cases)
                questions_text += examples_text
            else:
                questions_text += "\n(No similar examples found for this question)\n"
            
            questions_text += "\n---\n\n"

        questions_text += "Please generate model answers for all these Primary School Science questions using the examples provided as guidance for format and scientific accuracy. Pay attention to any sub-questions that may be related (like 29a, 29b, 29c) and ensure your answers are consistent across related questions."

        # Step 3: Call OpenAI API
        st.info("🤖 Generating answers with OpenAI...")
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": questions_text}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        ai_response = json.loads(response.choices[0].message.content)
        
        # Add metadata about skipped questions and similar cases used
        result_metadata = {
            "skipped_questions": skipped_questions,
            "similar_cases_used": {}
        }
        
        # Store information about which similar cases were used for each question
        for item in questions_with_examples:
            question_data = item['question_data']
            similar_cases = item['similar_cases']
            question_number = question_data.get('question_number', 'Unknown')
            
            if similar_cases:
                result_metadata["similar_cases_used"][str(question_number)] = [
                    {
                        "question": case.get('question', 'N/A')[:100] + "..." if len(case.get('question', '')) > 100 else case.get('question', 'N/A'),
                        "answer": case.get('answer', 'N/A')[:100] + "..." if len(case.get('answer', '')) > 100 else case.get('answer', 'N/A'),
                        "full_question": case.get('question', 'N/A'),
                        "full_answer": case.get('answer', 'N/A')
                    }
                    for case in similar_cases
                ]
            else:
                result_metadata["similar_cases_used"][str(question_number)] = []
        
        if skipped_questions:
            ai_response["skipped_questions"] = skipped_questions
            st.warning(f"⚠️ {len(skipped_questions)} questions were flagged due to issues with embedding or similar case retrieval.")
        
        # Add the similar cases metadata to the response
        ai_response["metadata"] = result_metadata
        
        return ai_response
        
    except Exception as e:
        st.error(f"Error generating batch answers: {e}")
        return {"error": str(e)}

def check_specific_paper(supabase_client: Client, paper_code: str) -> dict:
    """Check if a specific paper exists in the database and return details."""
    if not supabase_available or not supabase_client:
        return {"error": "Supabase connection not available"}
    
    try:
        # Query for the specific paper
        response = supabase_client.table(TARGET_TABLE) \
                                  .select("*") \
                                  .eq("paper", paper_code) \
                                  .execute()
        
        result = {
            "paper_code": paper_code,
            "exists": len(response.data) > 0,
            "total_rows": len(response.data) if response.data else 0,
            "data": response.data[:5] if response.data else []  # Show first 5 rows as sample
        }
        
        if response.data:
            # Get question types breakdown
            question_types = {}
            for row in response.data:
                qtype = row.get('question_type', 'Unknown')
                question_types[qtype] = question_types.get(qtype, 0) + 1
            result["question_types"] = question_types
            
            # Check for non-MCQ questions
            non_mcq = [row for row in response.data if row.get('question_type') not in ['MCQ', 'multiple_choice']]
            result["non_mcq_count"] = len(non_mcq)
        
        return result
        
    except Exception as e:
        return {"error": f"Database query failed: {e}"}
        

def format_similar_cases_as_json_examples(similar_cases: list[dict]) -> str:
    """Format similar cases as JSON examples for the OpenAI prompt."""
    if not similar_cases:
        return ""
    
    examples_text = "\n\nHere are some similar examples from the database to guide your answer format:\n\n"
    
    for i, case in enumerate(similar_cases, 1):
        try:
            question = case.get('question', 'N/A')
            model_answer = case.get('answer', 'N/A')
            astructure_str = case.get('astructure', '{}')
            
            # Parse the structured answer if it's a JSON string
            if isinstance(astructure_str, str):
                try:
                    structured_answer = json.loads(astructure_str)
                except json.JSONDecodeError:
                    structured_answer = {}
            else:
                structured_answer = astructure_str if astructure_str else {}
            
            example_json = {
                "model_answer": model_answer,
                "structured_answer": structured_answer
            }
            
            examples_text += f"Example {i}:\n"
            examples_text += f"Question: {question}\n"
            examples_text += f"Answer Format: {json.dumps(example_json, indent=2)}\n\n"
            
        except Exception as e:
            st.warning(f"Error formatting similar case {i}: {e}")
            continue
    
    return examples_text

def main():
    st.title("Generate Model Answers from Database Questions")

    if not supabase_available:
        st.error(f"Supabase connection to {selected_env} is not configured. Please set the required environment variables.")
        return
    if not openai_available:
        st.error("OpenAI client is not configured. Please set OPENAI_API_KEY environment variable.")
        return 

      
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Paper List"):
        # Clear any potential caches - use correct Streamlit method
        try:
            st.cache_data.clear()
        except:
            pass
        try:
            st.cache_resource.clear()
        except:
            pass
        st.rerun()    
  
    # Paper Code Input Section
    st.subheader("Enter Paper Code")
    
    # Text input for paper code
    paper_code_input = st.text_input(
        "Enter the paper code to process:",
        placeholder="e.g., P62024NHPSPL",
        help="Enter the exact paper code as it appears in the database"
    )
    
    # Validate paper code when entered
    paper_validation_result = None
    if paper_code_input and paper_code_input.strip():
        paper_code = paper_code_input.strip()
        
        # Check if paper exists
        paper_validation_result = check_specific_paper(supabase, paper_code)
        
        if "error" in paper_validation_result:
            st.error(f"❌ Database error: {paper_validation_result['error']}")
        elif not paper_validation_result['exists']:
            st.error(f"❌ Paper '{paper_code}' not found in the database.")
            st.info("💡 Make sure you've selected the correct environment (QA/PROD) and entered the exact paper code.")
        else:
            # Paper exists - show details
            st.success(f"✅ Paper '{paper_code}' found in database!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Questions", paper_validation_result['total_rows'])
            with col2:
                st.metric("Non-MCQ Questions", paper_validation_result.get('non_mcq_count', 0))
            with col3:
                mcq_count = paper_validation_result.get('question_types', {}).get('multiple_choice', 0)
                st.metric("MCQ Questions", mcq_count)
            
            # Show question type breakdown
            if paper_validation_result.get('question_types'):
                st.write("**Question Types:**")
                for qtype, count in paper_validation_result['question_types'].items():
                    st.write(f"- {qtype}: {count}")

    st.markdown("---")

    # Process the paper if it's valid
    if paper_validation_result and paper_validation_result.get('exists') and paper_code_input.strip():
        selected_paper = paper_code_input.strip()
        st.write(f"Processing Paper: **{selected_paper}**")
        
        # Initialize session state for storing processed data
        if 'generated_answers_data' not in st.session_state:
             st.session_state.generated_answers_data = None

        # Fetch and display questions for the selected paper
        st.subheader("Non-MCQ Questions for Selected Paper")
        questions_to_process = fetch_non_mcq_questions(supabase, selected_paper)
        
        if questions_to_process:
            # Display the questions in a dataframe
            questions_df = pd.DataFrame(questions_to_process)
            st.info(f"Found {len(questions_to_process)} non-MCQ questions.")
            st.dataframe(questions_df, height=400)
            
            # Show generate answers button
            if st.button(f"🤖 Generate Model Answers for {selected_paper}"):
                total_questions = len(questions_to_process)
                st.info(f"Generating model answers for {total_questions} questions using batch processing...")

                progress_bar = st.progress(0)
                status_text = st.empty()
                generated_data_list = []
                
                # Use batch processing instead of individual question processing
                status_text.text("Calling OpenAI for batch processing...")
                progress_bar.progress(0.1)
                
                # Generate answers for all questions at once
                ai_result = generate_answers_batch_with_openai(questions_to_process)
                progress_bar.progress(0.8)
                
                status_text.text("Processing AI response...")
                
                if "error" not in ai_result:
                    questions_answers = ai_result.get("questions_answers", {})
                    skipped_questions = ai_result.get("skipped_questions", [])
                    metadata = ai_result.get("metadata", {})
                    similar_cases_used = metadata.get("similar_cases_used", {})
                    success_count = 0
                    fail_count = 0
                    
                    # Show information about skipped questions
                    if skipped_questions:
                        st.warning(f"⚠️ {len(skipped_questions)} questions were skipped during processing:")
                        for skipped in skipped_questions:
                            st.write(f"- Question {skipped['question_number']}: {skipped['reason']}")
                    
                    # Show information about similar cases used
                    if similar_cases_used:
                        st.info("📚 Similar cases were found and used as examples for answer generation.")
                        with st.expander("View Similar Cases Used (Click to expand)"):
                            for qn, cases in similar_cases_used.items():
                                if cases:
                                    st.write(f"**Question {qn}** used {len(cases)} similar examples:")
                                    for i, case in enumerate(cases, 1):
                                        st.write(f"  {i}. Q: {case['question']}")
                                        st.write(f"     A: {case['answer']}")
                                        st.write("---")
                                else:
                                    st.write(f"**Question {qn}**: No similar examples found")
                    
                    # Debug: Show what question keys were returned by AI
                    st.info(f"📊 AI returned answers for keys: {list(questions_answers.keys())}")
                    
                    # Process each question result
                    for question_data in questions_to_process:
                        paper = question_data.get("paper")
                        qn = question_data.get("question_number")
                        question_text = question_data.get("question")
                        
                        # Attempt to derive level from paper_code
                        derived_level = None
                        if paper and len(paper) >= 2 and paper.startswith("P") and paper[1].isdigit():
                            potential_level = paper[:2]
                            if potential_level in ["P3", "P4", "P5", "P6"]:
                                derived_level = potential_level

                        current_result = {
                            "level": question_data.get('level', derived_level),
                            "paper": paper,
                            "question_number": qn,
                            "question": question_text,
                            "original_answer": question_data.get('answer', ''),
                            "marks": question_data.get('marks'),
                            "question_type": question_data.get('question_type'),  # Will be replaced by GPT classification if successful
                            "generated_answer": "",
                            "structured_answer": "",
                            "status": "failed",
                            "similar_cases_used": similar_cases_used.get(str(qn), [])
                        }

                        # Look for this question's answer in the batch response
                        question_answer = None
                        # Try different possible keys for the question number
                        possible_keys = [str(qn), f"question_{qn}", f"question_number_{qn}", qn]
                        
                        for key in possible_keys:
                            if key in questions_answers:
                                question_answer = questions_answers[key]
                                break
                        
                        if question_answer and isinstance(question_answer, dict):
                            model_answer = question_answer.get("model_answer", "")
                            structured_answer = question_answer.get("structured_answer", {})
                            gpt_question_type = question_answer.get("question_type", "")
                            
                            current_result.update({
                                "generated_answer": model_answer,
                                "structured_answer": json.dumps(structured_answer),
                                "question_type": gpt_question_type,  # Replace original with GPT classification
                                "status": "success"
                            })
                            success_count += 1
                        else:
                            current_result["generated_answer"] = f"Error: No answer found in batch response for question {qn}"
                            fail_count += 1
                        
                        generated_data_list.append(current_result)

                    progress_bar.progress(1.0)
                    status_text.text("Batch processing complete.")
                    
                    # Final Summary
                    st.success(f"Finished batch processing {selected_paper}.")
                    st.write(f"- Successfully generated answers: {success_count}")
                    st.write(f"- Failed to generate answers: {fail_count}")
                    
                    # Show question type distribution from GPT classification
                    if success_count > 0:
                        question_type_counts = {}
                        for result in generated_data_list:
                            if result["status"] == "success" and result.get("question_type"):
                                qtype = result["question_type"]
                                question_type_counts[qtype] = question_type_counts.get(qtype, 0) + 1
                        
                        if question_type_counts:
                            st.info("📊 **GPT Question Type Classification:**")
                            for qtype, count in question_type_counts.items():
                                st.write(f"  - **{qtype}**: {count} questions")
                    
                    if fail_count > 0:
                        st.warning(f"⚠️ {fail_count} questions failed. This may be due to question number formatting in the AI response.")
                        st.info("💡 You can review the results below and manually check any failed questions.")
                    
                else:
                    # Handle batch processing error
                    st.error(f"❌ Batch processing failed: {ai_result['error']}")
                    st.info("🔄 Consider trying again or checking your OpenAI API connection.")
                    
                    # Create error results for all questions
                    for question_data in questions_to_process:
                        paper = question_data.get("paper")
                        qn = question_data.get("question_number")
                        question_text = question_data.get("question")
                        
                        # Attempt to derive level from paper_code
                        derived_level = None
                        if paper and len(paper) >= 2 and paper.startswith("P") and paper[1].isdigit():
                            potential_level = paper[:2]
                            if potential_level in ["P3", "P4", "P5", "P6"]:
                                derived_level = potential_level

                        current_result = {
                            "level": question_data.get('level', derived_level),
                            "paper": paper,
                            "question_number": qn,
                            "question": question_text,
                            "original_answer": question_data.get('answer', ''),
                            "marks": question_data.get('marks'),
                            "question_type": question_data.get('question_type'),
                            "generated_answer": f"Batch Error: {ai_result['error']}",
                            "structured_answer": "",
                            "status": "failed",
                            "similar_cases_used": []  # Empty since batch processing failed
                        }
                        generated_data_list.append(current_result)

                # Store results in session state
                if generated_data_list:
                     st.session_state.generated_answers_data = pd.DataFrame(generated_data_list)
                     
                     # Also populate the review_data and extracted_data for compatibility with other pages
                     review_data_list = []
                     extracted_data_list = []
                     
                     for result in generated_data_list:
                         if result["status"] == "success":
                             review_data_list.append({
                                 'level': result['level'],
                                 'paper': result['paper'],
                                 'question_number': result['question_number'],
                                 'question': result['question'],
                                 'answer': result['generated_answer'],
                                 'marks': result['marks'],
                                 'question_type': result['question_type'],
                                 'structured_answer': result['structured_answer'],
                                 'evaluation_score': None,
                                 'evaluation_status': None,
                                 'evaluation_issues': None,
                                 'evaluation_suggested_fix': None,
                                 'review_status': 'pending',
                                 'revision_count': 0,
                                 'last_revision': datetime.now().isoformat(),
                                 'revision_error': None
                             })
                             
                             extracted_data_list.append({
                                 'level': result['level'],
                                 'paper': result['paper'],
                                 'question_number': result['question_number'],
                                 'question': result['question'],
                                 'answer': result['generated_answer'],
                                 'marks': result['marks'],
                                 'question_type': result['question_type'],
                                 'astructure': result['structured_answer']
                             })
                     
                     st.session_state.review_data = pd.DataFrame(review_data_list)
                     st.session_state.extracted_data = pd.DataFrame(extracted_data_list)
                else:
                     st.session_state.generated_answers_data = None
        else:
            st.warning("No non-MCQ questions found for this paper.")

    # --- Display Generated Results ---
    if st.session_state.get('generated_answers_data') is not None:
         st.markdown("---")
         st.subheader("Generated Model Answers")
         st.info("Review the generated answers below. The table includes both original question types and GPT's classification.")

         results_df = st.session_state.generated_answers_data
         
         # Add a helpful explanation of the columns
         st.write("**Column Guide:**")
         st.write("- `question_type`: GPT's automatic classification (fact_recall, explanation, experimental_purpose, range)")
         st.write("- Original database question types are replaced with GPT's intelligent classification")
         
         st.dataframe(results_df, height=600)
         
         # Add detailed view of similar cases used
         st.subheader("📚 Multi-Shot Examples Used")
         st.info("Below you can see which similar questions were used as examples for each generated answer.")
         
         for idx, row in results_df.iterrows():
             question_number = row['question_number']
             similar_cases = row.get('similar_cases_used', [])
             
             with st.expander(f"Question {question_number}: {row['question_type'].title()} ({len(similar_cases)} examples)"):
                 if similar_cases:
                     st.write(f"**Original Question:** {row['question']}")
                     st.write(f"**Generated Answer:** {row['generated_answer']}")
                     st.write("---")
                     st.write("**Similar Examples Used:**")
                     
                     for i, case in enumerate(similar_cases, 1):
                         st.write(f"**Example {i}:**")
                         st.write(f"📝 **Question:** {case.get('full_question', 'N/A')}")
                         st.write(f"✅ **Answer:** {case.get('full_answer', 'N/A')}")
                         if i < len(similar_cases):
                             st.write("---")
                 else:
                     st.write(f"**Question:** {row['question']}")
                     st.write(f"**Generated Answer:** {row['generated_answer']}")
                     st.warning("⚠️ No similar examples were used for this question (missing embeddings or vector search failed)")
         
         # Navigation to review page
         if not st.session_state.review_data.empty:
             if st.button("📝 Go to Review and Re-answer Page"):
                 st.switch_page("pages/2_Review_and_Reanswer.py")

# --- Run the app ---
if __name__ == "__main__":
    main()

