import streamlit as st
from pathlib import Path
import sys
import os

# Add the project root to the Python path
# This is necessary to import the notebook_runner utility
# Assumes this script is in dashboard/pages/
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from src.utils.notebook_runner import NotebookRunner

# --- Page Configuration ---
st.set_page_config(
    page_title="Notebook Explorer",
    page_icon="https://i.imgur.com/6fJp7ss.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Font Awesome CSS ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css">
""", unsafe_allow_html=True)


# --- Styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E3B4E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)

# --- Page Content ---
st.markdown('<p class="main-header"><i class="fa-solid fa-book-open"></i> Notebook Explorer</p>', unsafe_allow_html=True)
st.info(
    "**Welcome to the Notebook Explorer!** Here you can run and view the results of various analytical notebooks "
    "directly within the dashboard. Select a notebook, adjust its parameters if available, and click 'Run Notebook' "
    "to see the analysis."
)

# --- Initialize Notebook Runner ---
notebooks_dir = root_path / "notebooks"
runner = NotebookRunner(notebooks_dir)
available_notebooks = runner.list_notebooks()

# --- Sidebar for Notebook Selection ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    # Create a mapping from display name to file path
    notebook_display_names = {v['name']: k for k, v in available_notebooks.items()}
    
    selected_notebook_name = st.selectbox(
        "**Select a Notebook**",
        options=list(notebook_display_names.keys())
    )
    
    selected_notebook_path = notebook_display_names.get(selected_notebook_name)

    if selected_notebook_path:
        st.markdown(f"**Description:** {available_notebooks[selected_notebook_path]['description']}")

        # --- Dynamic Parameter Form ---
        with st.form(key="notebook_params_form"):
            st.subheader("Parameters")
            params = runner.extract_parameters(selected_notebook_path)
            
            if not params:
                st.write("This notebook has no configurable parameters.")
            
            injected_params = runner.create_parameter_form(params)
            
            run_button = st.form_submit_button(label="🚀 Run Notebook")

# --- Main Panel for Displaying Results ---
if selected_notebook_path:
    st.markdown(f"## 🔬 Results for `{selected_notebook_name}`")
    
    if run_button:
        with st.spinner(f"Executing {selected_notebook_name}... This may take a moment."):
            try:
                # Execute the notebook and get the path to the HTML report
                report_path, output_notebook_path = runner.execute_notebook(selected_notebook_path, injected_params)
                
                # Store paths in session state to persist them
                st.session_state['report_path'] = report_path
                st.session_state['output_notebook_path'] = output_notebook_path
                st.session_state['last_run_notebook'] = selected_notebook_path
                
                st.success("✅ Notebook executed successfully!")

            except Exception as e:
                st.error(f"🔥 An error occurred while running the notebook: {e}")
                st.exception(e)
                st.session_state['report_path'] = None
                st.session_state['output_notebook_path'] = None

    # --- Display the results if a report exists ---
    if 'report_path' in st.session_state and st.session_state.get('last_run_notebook') == selected_notebook_path:
        report_path = st.session_state['report_path']
        output_notebook_path = st.session_state['output_notebook_path']

        if report_path and os.path.exists(report_path):
            # --- Extract and display key results (plots, dataframes) ---
            st.markdown("---")
            st.subheader("✨ Key Outputs")
            
            with st.spinner("Extracting key outputs from the notebook..."):
                try:
                    results = runner.extract_results(output_notebook_path)
                    
                    if not results:
                        st.write("No specific outputs (plots, dataframes) were detected in the notebook.")
                    
                    for result in results:
                        if result['type'] == 'plot':
                            st.plotly_chart(result['content'], use_container_width=True)
                        elif result['type'] == 'dataframe':
                            st.dataframe(result['content'])
                        elif result['type'] == 'markdown':
                            st.markdown(result['content'], unsafe_allow_html=True)
                        elif result['type'] == 'text':
                            st.code(result['content'], language='text')
                
                except Exception as e:
                    st.warning(f"Could not extract key outputs: {e}")

            with st.expander("📊 View Full HTML Report"):
                with open(report_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)

        else:
            st.info("Run the notebook to generate a report.")
else:
    st.info("Select a notebook from the sidebar to get started.")
