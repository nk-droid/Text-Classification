import streamlit as st

import tensorflow as tf
import numpy as np
from keras.models import load_model

@st.cache_resource
def load_model_():
    model = load_model("./models/disaster_tweet_classifier")
    return model
with st.spinner('Resources are being loaded...'):
  model=load_model_()
 
st.write("""
         # Disaster Tweet Classifier
         """)

tweet = st.text_input('Provide the text for prediction')

def predict(tweet, model):
    
    return model.predict(np.array([tweet]))
    
if st.button("Predict"):
    if tweet == "":
        st.exception(RuntimeWarning("Please provide some text to classify."))
    else:
        predicted_class = predict(tweet, model)
        if predicted_class > 0.5:
            st.error("The text is related to disaster.")
        else:
            st.success("The text isn't related to disaster.")