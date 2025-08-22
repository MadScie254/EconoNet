import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="EconoNet Test",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 EconoNet Test Dashboard")

st.success("✅ All systems operational!")
st.info("This is a test version to verify the dashboard works.")

# Simple test data
test_data = pd.DataFrame({
    'Date': pd.date_range('2020-01-01', periods=12, freq='M'),
    'Value': np.random.randn(12).cumsum() + 100
})

st.line_chart(test_data.set_index('Date'))

st.markdown("### 🎯 Dashboard Features Working")
st.write("- ✅ Basic Streamlit functionality")
st.write("- ✅ Data visualization")
st.write("- ✅ Pandas integration")
st.write("- ✅ NumPy calculations")

if st.button("🧪 Run Tests"):
    st.balloons()
    st.success("All tests passed! 🎉")
