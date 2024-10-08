import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

# Sample data based on the provided graph
features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
chi2_scores = [0.5, 1.0, 0.2, 3.5, 0.1, 24.0, 0.3]
p_values = [0.6, 0.7, 0.8, 0.1, 0.7, 0.01, 0.75]

# Create a DataFrame
data = pd.DataFrame({
    'Feature': features,
    'Chi-Square Score': chi2_scores,
    'P-Value': p_values
})

# Streamlit app
st.title("Chi-Square Test Interactive App")

st.write("""
### Explore Chi-Square Scores and P-Values
Click on a feature to see its chi-square score and p-value.
""")

# Sidebar for feature selection with a multiselect option
selected_features = st.sidebar.multiselect("Select features", features, default=features)

# Filter data based on selected features
filtered_data = data[data['Feature'].isin(selected_features)]

# Display selected features' chi-square scores and p-values in a table
st.write("**Selected Features Data**")
st.dataframe(filtered_data)

# Plotting the bar charts with tooltips
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Chi-Square Scores plot
ax[0].bar(filtered_data['Feature'], filtered_data['Chi-Square Score'], color='skyblue')
ax[0].set_title('Chi-Square Scores')
ax[0].set_ylabel('Score')
ax[0].tick_params(axis='x', rotation=45)

# P-Values plot
ax[1].bar(filtered_data['Feature'], filtered_data['P-Value'], color='salmon')
ax[1].set_title('P-Values')
ax[1].set_ylabel('P-Value')
ax[1].tick_params(axis='x', rotation=45)

st.pyplot(fig)

# Function to perform label encoding
def label_encode(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

# Streamlit app
st.title("Chi-Square Test Interactive App")

st.write("""
### Upload Your Dataset
Upload a CSV file or provide a link to perform chi-square analysis.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# URL input
url_input = st.text_input("Or enter a URL to a CSV file")

# Load data from file or URL
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif url_input:
    try:
        df = pd.read_csv(url_input)
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        df = None
else:
    df = None

if df is not None:
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Select columns for chi-square calculation
    target_column = st.selectbox("Select the target column", df.columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in df.columns if col != target_column])

    if target_column and feature_columns:
        # Encode categorical variables
        df_encoded = label_encode(df[[target_column] + feature_columns])

        # Perform chi-square test
        X = df_encoded[feature_columns]
        y = df_encoded[target_column]
        chi2_scores, p_values = chi2(X, y)

        # Create DataFrame for results
        results_df = pd.DataFrame({
            'Feature': feature_columns,
            'Chi-Square Score': chi2_scores,
            'P-Value': p_values
        })

        st.write("### Chi-Square Analysis Results")
        st.dataframe(results_df)

        # Plotting the bar charts
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Chi-Square Scores plot
        ax[0].bar(results_df['Feature'], results_df['Chi-Square Score'], color='skyblue')
        ax[0].set_title('Chi-Square Scores')
        ax[0].set_ylabel('Score')
        ax[0].tick_params(axis='x', rotation=45)

        # P-Values plot
        ax[1].bar(results_df['Feature'], results_df['P-Value'], color='salmon')
        ax[1].set_title('P-Values')
        ax[1].set_ylabel('P-Value')
        ax[1].tick_params(axis='x', rotation=45)

        st.pyplot(fig)