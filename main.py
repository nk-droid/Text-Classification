import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Text Classifier"
    )

    st.write("# Overview")

    st.sidebar.success("Select a text classifier above.")

    st.markdown(
        """
        This project contains a text classification model.
        
        ### Disaster Tweet Classifier

    """
    )


if __name__ == "__main__":
    run()