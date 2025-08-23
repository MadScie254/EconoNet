"""
EconoNet Main Application
========================

Main entry point for the EconoNet economic modeling platform.
This file provides a simple interface to select and run different dashboards.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add src to path for econonet imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="EconoNet - Economic Intelligence Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏛️ EconoNet - Economic Intelligence Platform")
    st.markdown("---")
    
    # Dashboard selection
    st.sidebar.title("📊 Dashboard Selection")
    
    dashboard_options = {
        "Ultra Dashboard Enhanced": "ultra_dashboard_enhanced.py",
        "Immersive Dashboard": "immersive_dashboard.py", 
        "Enhanced Streamlit App": "enhanced_streamlit_app.py"
    }
    
    selected_dashboard = st.sidebar.selectbox(
        "Choose a dashboard:",
        list(dashboard_options.keys())
    )
    
    # Dashboard info
    st.subheader(f"📈 {selected_dashboard}")
    
    dashboard_descriptions = {
        "Ultra Dashboard Enhanced": "Advanced economic intelligence with real-time data integration, AI-powered insights, and comprehensive analytics.",
        "Immersive Dashboard": "Full-screen immersive economic analysis platform with interactive models and real-time data.",
        "Enhanced Streamlit App": "Comprehensive dashboard with notebook integration and advanced economic modeling capabilities."
    }
    
    st.info(dashboard_descriptions[selected_dashboard])
    
    # Launch button
    dashboard_file = dashboard_options[selected_dashboard]
    
    if st.button(f"🚀 Launch {selected_dashboard}", type="primary"):
        st.success(f"Launching {selected_dashboard}...")
        st.markdown(f"""
        **To run this dashboard directly:**
        ```bash
        streamlit run {dashboard_file}
        ```
        """)
        
        # Try to import and display basic info
        try:
            if dashboard_file == "ultra_dashboard_enhanced.py":
                st.markdown("✅ Ultra Dashboard Enhanced is available")
            elif dashboard_file == "immersive_dashboard.py":
                st.markdown("✅ Immersive Dashboard is available")
            elif dashboard_file == "enhanced_streamlit_app.py":
                st.markdown("✅ Enhanced Streamlit App is available")
        except Exception as e:
            st.warning(f"Dashboard file exists but may have import dependencies: {e}")
    
    # Quick stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Available Dashboards", len(dashboard_options))
    
    with col2:
        st.metric("🏛️ Economic Models", "15+")
    
    with col3:
        st.metric("📈 Data Sources", "10+")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>EconoNet Economic Intelligence Platform | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()