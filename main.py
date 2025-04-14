import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model dan tokenizer
try:
    model = load_model("model_mental_health_v2x.h5")
    
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    # Parameter input
    max_len = 100  # sesuaikan dengan model kamu
    
    # Label
    labels = ['Anxiety', 'Depression']  # sesuaikan dengan output model - updated to match your previous model
    
    # Streamlit UI
    st.title("Mental Health Detection")
    st.write("Share how you're feeling and we'll analyze your mental health state.")
    
    text_input = st.text_area("How are you feeling today?", height=150)
    
    if st.button("Analyze"):
        if text_input:
            with st.spinner("Analyzing your response..."):
                # Preprocessing
                sequences = tokenizer.texts_to_sequences([text_input])
                padded = pad_sequences(sequences, maxlen=max_len)
                
                # Prediksi
                prediction = model.predict(padded)
                predicted_label = labels[np.argmax(prediction)]
                confidence = float(np.max(prediction)) * 100
                
                # Hasil
                st.subheader("Analysis Results")
                st.write(f"Mental Health Condition: **{predicted_label}**")
                st.write(f"Confidence: **{confidence:.2f}%**")
                
                # Display probabilities
                st.subheader("Probability Breakdown:")
                for i, label in enumerate(labels):
                    prob = float(prediction[0][i]) * 100
                    st.write(f"- {label}: {prob:.2f}%")
                    st.progress(int(prob))
                
                # Disclaimer
                st.info("Note: This is not a medical diagnosis. If you're experiencing mental health difficulties, please consult with a healthcare professional.")
        else:
            st.warning("Please enter some text before analyzing.")

except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.write("Please make sure your model and tokenizer files are in the correct location and format.")