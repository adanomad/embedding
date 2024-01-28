import streamlit as st
import pandas as pd


def main():
    st.title("CSV File Browser")

    # Allow user to upload a CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)

        # Display the dataframe
        st.write(data)

        # Show some basic statistics about the dataframe
        if st.checkbox("Show summary"):
            st.write(data.describe())

        # Allow the user to select a subset of columns to display
        if st.checkbox("Select columns to display"):
            selected_columns = st.multiselect("Select columns", data.columns)
            if selected_columns:
                st.write(data[selected_columns])


if __name__ == "__main__":
    main()
