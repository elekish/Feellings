import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis', framework='pt')

# Load pre-trained GPT-2 model and tokenizer with explicit framework
gpt2_model_name = "gpt2"  # You can replace this with a fine-tuned model if available

# Specify the framework you want to use (e.g., 'pt' for PyTorch or 'tf' for TensorFlow)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name, framework='pt')

# Streamlit app
st.title("E-Therapist")

user_input = st.text_input("Enter your message:")
if st.button("Submit"):
    # Analyze the sentiment of the user input
    sentiment_result = sentiment_analyzer(user_input)
    predicted_sentiment = sentiment_result[0]['label']

    # Generate a response based on sentiment
    input_text = f"Sentiment: {predicted_sentiment}\nTherapist:"
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")

    # Generate a response
    output = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    response = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    st.subheader("Therapist's Response:")
    st.write(response)
