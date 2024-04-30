import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
@st.cache
def load_data():
    # Load your dataset here, replace 'data.csv' with your actual data file
    df = pd.read_csv('data/advfextract.csv')
    return df

def main():
    st.title("Exploratory Data Analysis")
    # Display data visualization
    st.sidebar.title("Table of Contents")
    options = ["Dataset Overview", "Basic Statistics", "Histograms", "Violin Plots", "Line Plots", "Scatter Plots"]
    choice = st.sidebar.radio("Go to", options)

    # Load the data
    df = load_data()

    # Display the first few rows of the dataset
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(df.describe())

    # Display data visualization
    st.subheader("Data Visualization")

    if choice == "Histograms":
        # Histograms for numerical columns
        st.subheader("Histograms for Numerical Columns")
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[column], kde=True)
            plt.title(f"Histogram of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            st.pyplot(plt)
    elif choice == "Violin Plots":
        # Violin plots for numerical columns
        st.subheader("Violin Plots for Numerical Columns")
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            plt.figure(figsize=(8, 6))
            sns.violinplot(y=df[column])
            plt.title(f"Violin plot of {column}")
            plt.ylabel(column)
            st.pyplot(plt)
    elif choice == "Line Plots":
        # Line plot for numerical columns
        st.subheader("Line Plot for Numerical Columns")
        for column in df.select_dtypes(include=['int64', 'float64']).columns:
            plt.figure(figsize=(8, 6))
            sns.lineplot(data=df, x=df.index, y=column)
            plt.title(f"Line plot of {column}")
            plt.xlabel("Index")
            plt.ylabel(column)
            st.pyplot(plt)


if __name__ == "__main__":
    main()

