import streamlit as st
import pandas as pd
import json
import os
from supabase import create_client, Client

# --- Environment Selection ---
ENV_OPTIONS = ["QA", "PROD"]
selected_env = st.sidebar.selectbox("Select Environment", ENV_OPTIONS, index=0)

# --- Supabase Initialization ---
if selected_env == "QA":
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")
elif selected_env == "PROD":
    supabase_url = os.getenv("SUPABASEO1_URL")
    supabase_key = os.getenv("SUPABASEO1_API_KEY")
else: # Default to QA if something goes wrong, though selectbox should prevent this
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")
    st.warning(f"Unknown environment selected: {selected_env}. Defaulting to QA.")

supabase: Client | None = None
supabase_available = False

try:
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        supabase_available = True
    else:
        st.warning(f"Supabase URL or Key not found for {selected_env} environment in environment variables. Saving disabled.")
except Exception as e:
    st.error(f"Failed to initialize Supabase client: {e}")
# ---------------------------

# --- Authentication Check ---
if st.session_state.get("authenticated", False):

    st.title("View and Edit Extracted Questions")

    # Load structured output if it exists
    if "extracted_data" in st.session_state and isinstance(st.session_state.extracted_data, pd.DataFrame):
        st.info("You can edit the data below before saving.")
        # Use data editor for modifications
        edited_df = st.data_editor(
            st.session_state.extracted_data,
            num_rows="dynamic", # Allow adding/deleting rows
            key="data_editor"
        )

        # --- Column Selection for Update ---
        if edited_df is not None and not edited_df.empty:
            all_columns = edited_df.columns.tolist()
            # Exclude known key columns and non-data columns like page_number
            # Assuming 'paper' and 'question_number' are primary keys and always needed for matching.
            # 'page_number' is explicitly dropped later, so no need to offer it for update.
            updatable_columns = [
                col for col in all_columns 
                if col not in ['paper', 'question_number', 'page_number']
            ]
            
            if not updatable_columns:
                st.info("No other columns available to select for update besides key columns.")
                selected_columns_to_update = []
            else:
                selected_columns_to_update = st.multiselect(
                    "Select columns to update in Supabase (besides keys 'paper' & 'question_number'):",
                    options=updatable_columns,
                    default=updatable_columns # Default to all updatable columns
                )
        else:
            selected_columns_to_update = []
        # ------------------------------------

        # --- Duplicate Row Check ---
        if edited_df is not None and not edited_df.empty and \
           'paper' in edited_df.columns and 'question_number' in edited_df.columns:
            
            # Identify all rows that are part of any duplicate set for 'paper' and 'question_number'
            # keep=False marks all duplicates as True
            duplicates = edited_df.duplicated(subset=['paper', 'question_number'], keep=False)
            
            if duplicates.any():
                st.warning("Duplicate Paper and Question Number combinations found! Please resolve before saving:")
                # Select the duplicate rows and only show relevant columns for the warning
                duplicate_rows_to_show = edited_df[duplicates][['paper', 'question_number']].copy()
                # To avoid issues with Streamlit trying to index with duplicate index values from the original df,
                # we reset the index for display purposes or convert to a list of strings.
                # For simplicity, let's list them.
                # Group by paper and question_number to list each duplicated combination once in the warning summary.
                grouped_duplicates = duplicate_rows_to_show.groupby(['paper', 'question_number']).size().reset_index(name='count')
                for _, row in grouped_duplicates.iterrows():
                    st.markdown(f"  - Paper: **{row['paper']}**, Question Number: **{row['question_number']}** (appears {row['count']} times)")
        # ---------------------------

        # Check if Supabase is available before showing the button
        if supabase_available:
            if st.button("Save to Supabase (pri_sci_paper table)"):
                if edited_df is not None and not edited_df.empty:
                    with st.spinner("Saving data to Supabase..."):
                        try:
                            # --- Exclude page_number before upserting ---
                            df_to_save = edited_df.copy()
                            if 'page_number' in df_to_save.columns:
                                df_to_save = df_to_save.drop(columns=['page_number'])
                            # ----------------------------------------------
                            
                            # Convert DataFrame to list of dictionaries for upsert,
                            # including only selected columns + key columns
                            data_to_upsert = []
                            if not df_to_save.empty:
                                for index, row in df_to_save.iterrows():
                                    record = {}
                                    # Always include key columns
                                    if 'paper' in row:
                                        record['paper'] = row['paper']
                                    if 'question_number' in row:
                                        record['question_number'] = row['question_number']
                                    
                                    # Include selected columns for update
                                    for col in selected_columns_to_update:
                                        if col in row:
                                            record[col] = row[col]
                                    
                                    # Only add if key columns are present
                                    if 'paper' in record and 'question_number' in record:
                                         data_to_upsert.append(record)
                                    else:
                                        st.warning(f"Skipping row {index+1} due to missing 'paper' or 'question_number'.")
                            
                            if not data_to_upsert:
                                st.warning("No data to save after filtering for keys and selected columns.")
                                # Use 'continue' if in a loop or 'return' if in a function, 
                                # for now, this will prevent the upsert call if data_to_upsert is empty.
                            else:
                                # Ensure Supabase client is available (double check)
                                if supabase:
                                    # Perform upsert operation with filtered data
                                    response = supabase.table("pri_sci_paper").upsert(data_to_upsert).execute()
                                    
                                    # Check response (basic check, specific API responses may vary)
                                    if response.data:
                                        st.success(f"Successfully saved/updated {len(response.data)} records in Supabase!")
                                        # Optionally refresh or clear data after saving
                                        # st.session_state.extracted_data = pd.DataFrame() # Example clear
                                    else:
                                        # Attempt to access error information if available (structure might vary)
                                        error_message = "Unknown error during upsert."
                                        if hasattr(response, 'error') and response.error:
                                            error_message = f"Supabase error: {response.error.message if hasattr(response.error, 'message') else response.error}"
                                        st.error(f"Failed to save data to Supabase. {error_message}")

                                else:
                                    st.error("Supabase client is not initialized.")

                        except Exception as e:
                            st.error(f"An error occurred during Supabase operation: {e}")
                else:
                    st.warning("No data to save or data is empty after editing.") # Updated warning
        elif not supabase_available:
             st.warning("Supabase is not configured. Cannot save data.")

    elif "extracted_data" in st.session_state: # Handle case where it exists but isn't a DataFrame
         st.error("Extracted data is not in the expected format (DataFrame).")
    else:
        st.warning("No extracted data found. Please upload and process a paper first on the 'Upload and Extract' page.")

else:
    st.warning("Please log in first to view or edit extracted questions.")
