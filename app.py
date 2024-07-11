import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('advice_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit app
st.title('Advice on How to Talk to Girls 💬❤️')
st.write('Enter your scenario below and get advice!')

user_input = st.text_area("Your Scenario:")

if st.button('Get Advice'):
    if user_input:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)
        st.write(f'Advice: {prediction[0]}')
    else:
        st.write('Please enter a scenario.')

# Styling the app
st.markdown("""
    <style>
    body {
        background-image: url("https://your-image-url.jpg");
        background-size: cover;
        font-family: Arial, Helvetica, sans-serif;
    }
    h1 {
        color: #333399;
        text-shadow: 2px 2px #ff99cc;
    }
    .stButton>button {
        background-color: #333399;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #ff99cc;
    }
    .stTextArea>div>div>textarea {
        background-color: #f0f0f5;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<div style='text-align: center;'>😊👋🌸💬</div>", unsafe_allow_html=True)
