"""
Exploratory Data Analysis (EDA) Page
======================================

This page allows users to upload a dataset and perform a comprehensive
Exploratory Data Analysis (EDA) on it.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.api_integration import DataCleaner

def perform_eda(df):
    """
    Performs EDA on the given dataframe and displays the results.
    """
    st.header("Exploratory Data Analysis")

    # Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Data Summary
    st.subheader("Data Summary")
    st.write(df.describe())

    # Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig)
    else:
        st.info("No numeric columns to create a correlation heatmap.")

    # Histograms for numeric columns
    st.subheader("Histograms")
    for col in numeric_cols:
        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        st.plotly_chart(fig)

    # Bar charts for categorical columns
    st.subheader("Bar Charts")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        fig = px.bar(df[col].value_counts(), title=f"Bar Chart of {col}")
        st.plotly_chart(fig)
        
    # Geographic Visualization
    st.subheader("Geographic Visualization")
    country_cols = [col for col in df.columns if 'country' in col.lower() or 'region' in col.lower()]
    if country_cols:
        country_col = country_cols[0]
        value_cols = numeric_cols
        if value_cols.any():
            value_col = value_cols[0]
            fig = px.choropleth(df, locations=country_col, locationmode='country names', color=value_col,
                                hover_name=country_col, color_continuous_scale=px.colors.sequential.Plasma,
                                title=f"{value_col} by Country")
            st.plotly_chart(fig)
        else:
            st.info("No numeric columns to visualize on the map.")
    else:
        st.info("No country or region column found for geographic visualization.")


def main():
    """Main function for the EDA page"""
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Upload a dataset to perform a comprehensive EDA.")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        data_cleaner = DataCleaner()
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            cleaned_df = data_cleaner.clean_dataframe(df)
            st.success("File uploaded and cleaned successfully!")
            
            perform_eda(cleaned_df)

        except Exception as e:
            st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
