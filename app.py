# import streamlit as st
# import pickle
# from sklearn.feature_extraction.text import TfidfVectorizer
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# # import nltk
# # nltk.download('stopwords')


# import nltk
# nltk.data.path.append('./nltk_data')
# from nltk.corpus import stopwords


# import os
# base_dir = os.path.dirname(__file__)
# model_path = os.path.join(base_dir, "model", "twitter_sentiment_trained_model.sav")
# vectorizer_path = os.path.join(base_dir, "vectorizer", "vectorizer.pkl")

# # Load the trained model and vectorizer
# loaded_model = pickle.load(open(model_path, 'rb'))
# vectorizer = pickle.load(open(vectorizer_path, 'rb'))  # Assuming you saved the vectorizer

# # Preprocessing function (same as before)
# port_stem = PorterStemmer()
# def stemming(content):
#   stemmed_content = re.sub('[^a-zA-Z]',' ',content)
#   stemmed_content = stemmed_content.lower()
#   stemmed_content = stemmed_content.split()
#   stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
#   stemmed_content = ' '.join(stemmed_content)
#   return stemmed_content

# # Streamlit app
# def main():
#     st.title("Twitter Sentiment Analysis")
    
#     # Input for new tweet
#     new_tweet = st.text_area("Enter a tweet:")
    
#     if st.button("Predict"):
#         # Preprocess the tweet
#         processed_tweet = stemming(new_tweet)
        
#         # Vectorize the tweet
#         vectorized_tweet = vectorizer.transform([processed_tweet])
        
#         # Make prediction
#         prediction = loaded_model.predict(vectorized_tweet)
        
#         # Display prediction
#         if prediction[0] == 0:
#             st.write("Prediction: Negative Tweet")
#         else:
#             st.write("Prediction: Positive Tweet")

# if __name__ == '__main__':
#     main()
    









import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import os

# Ensure the NLTK data path is correctly set
nltk.data.path.append('./nltk_data')

# Download stopwords if not already available locally
# Uncomment this line if stopwords data is not available
# nltk.download('stopwords', download_dir='./nltk_data')

# Initialize stopwords
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Paths to model and vectorizer
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "model", "twitter_sentiment_trained_model.sav")
vectorizer_path = os.path.join(base_dir, "vectorizer", "vectorizer.pkl")

# Load the trained model and vectorizer
try:
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Streamlit app
def main():
    st.title("Twitter Sentiment Analysis")

    # Input for new tweet
    new_tweet = st.text_area("Enter a tweet:")

    if st.button("Predict"):
        if not new_tweet:
            st.warning("Please enter a tweet to analyze.")
            return
        
        # Preprocess the tweet
        processed_tweet = stemming(new_tweet)

        # Vectorize the tweet
        vectorized_tweet = vectorizer.transform([processed_tweet])

        # Make prediction
        prediction = loaded_model.predict(vectorized_tweet)

        # Display prediction
        if prediction[0] == 0:
            st.write("Prediction: Negative Tweet")
        else:
            st.write("Prediction: Positive Tweet")

if __name__ == '__main__':
    main()
