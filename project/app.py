import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import joblib
import numpy as np
import os

# 1. Page Configuration
st.set_page_config(page_title="TRACT Classifier")

st.title(" TRACT: Tweet Analysis & Classification")
st.markdown("""
This system utilizes **BERT (Bidirectional Encoder Representations from Transformers)** 
to classify social media posts into categories based on their semantic context.
""")

# 2. Model Loading
@st.cache_resource
def load_models():
    # Load BERT Tokenizer and Model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Load your trained Logistic Regression classifier
    # Make sure 'bert_classifier.pkl' is in the same folder as this script
    if os.path.exists('bert_classifier.pkl'):
        classifier = joblib.load('bert_classifier.pkl')
    else:
        classifier = None
        st.error("Error: 'bert_classifier.pkl' not found. Please save your model first.")
    
    return tokenizer, bert_model, classifier

tokenizer, bert_model, classifier = load_models()

# 3. Feature Extraction Function
def get_bert_embedding(text):
    """
    Extracts the [CLS] token embedding from BERT which 
    serves as a summary of the entire input text.
    """
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        
    # Extract the CLS embedding (index 0)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 4. User Interface Inputs
st.subheader("Analyze a Tweet")
user_input = st.text_area(
    "Enter tweet text below:", 
    placeholder="e.g., Reporting a harassment incident near the station..."
)

if st.button("Classify Tweet"):
    if user_input and classifier is not None:
        with st.spinner('BERT is extracting semantic features...'):
            # Step 1: Extract Semantic Features (Embeddings)
            embedding = get_bert_embedding(user_input)
            
            # Step 2: Real Prediction using your trained model
            # We reshape the embedding because the model expects a 2D array
            prediction = classifier.predict(embedding.reshape(1, -1))[0]
            
            # 5. Displaying the Result
            st.divider()
            st.write("### Prediction Result:")
            
            # Map the numeric labels back to their meanings
            if prediction == 1:
                st.error(" **Category: Abuse (Incident Report)**")
                st.info("The model detected language indicative of a reporting incident.")
            elif prediction == 2:
                st.warning(" **Category: Empathy (Social Support)**")
                st.info("The model detected supportive or empathetic sentiment.")
            else:
                st.success(" **Category: General (Non-Critical)**")
                st.info("The model classified this as a general or neutral tweet.")
    elif classifier is None:
        st.error("Model file missing. Prediction cannot be performed.")
    else:
        st.warning("Please enter a text input to analyze.")

# Footer
st.markdown("---")
st.caption("Developed by: Asmaa | Faculty of Computers and AI - Damietta University")