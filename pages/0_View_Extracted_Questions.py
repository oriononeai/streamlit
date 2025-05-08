import streamlit as st
import pandas as pd
import json
import os
from supabase import create_client, Client

# --- Supabase Initialization ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_API_KEY")
supabase: Client | None = None
supabase_available = False

try:
    if supabase_url and supabase_key:
        supabase = create_client(supabase_url, supabase_key)
        supabase_available = True
    else:
        st.warning("Supabase URL or Key not found in environment variables. Saving disabled.")
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
        # Optionally store edited data back to session state if needed elsewhere immediately
        # st.session_state.extracted_data = edited_df

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
                            
                            # Convert filtered DataFrame to list of dictionaries for upsert
                            data_to_upsert = df_to_save.to_dict(orient="records")
                            
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
                    st.warning("No data to save.")
        elif not supabase_available:
             st.warning("Supabase is not configured. Cannot save data.")

    elif "extracted_data" in st.session_state: # Handle case where it exists but isn't a DataFrame
         st.error("Extracted data is not in the expected format (DataFrame).")
    else:
        st.warning("No extracted data found. Please upload and process a paper first on the 'Upload and Extract' page.")

else:
    st.warning("Please log in first to view or edit extracted questions.")
