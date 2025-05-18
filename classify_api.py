# classify_api.py
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob # Import TextBlob
import re
import string
import nltk
import os
import requests

nltk_data_path = os.environ.get("NLTK_DATA")
if nltk_data_path:
    nltk.data.path.insert(0, nltk_data_path) # Add custom path to NLTK search paths
    print(f"Added {nltk_data_path} to NLTK data search path.")
else:
    nltk_data_path = nltk.data.path[0]
    print(f"NLTK_DATA environment variable not set, using default download path: {nltk_data_path}")
print("Checking for NLTK resources for API script...")
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' found.")
except LookupError:
    print("NLTK 'punkt' not found. Attempting download during startup...")
    try:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True) # Pass download_dir
        print("NLTK 'punkt' downloaded.")
    except Exception as e:
        print(f"Error downloading 'punkt' during startup: {e}") # More specific error
try:
    nltk.data.find('corpora/stopwords')
    print("NLTK 'stopwords' found.")
except LookupError:
    print("NLTK 'stopwords' not found. Attempting download during startup...")
    try:
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        print("NLTK 'stopwords' downloaded.")
    except Exception as e:
        print(f"Error downloading 'stopwords' during startup: {e}")

try:
    nltk.data.find('corpora/wordnet')
    print("NLTK 'wordnet' found.")
except LookupError:
    print("NLTK 'wordnet' not found. Attempting download during startup...")
    try:
        nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
        print("NLTK 'wordnet' downloaded.")
    except Exception as e:
        print(f"Error downloading 'wordnet' during startup: {e}")

try:
    nltk.data.find('corpora/omw-1.4')
    print("NLTK 'omw-1.4' found.")
except LookupError:
    print("NLTK 'omw-1.4' not found. Attempting download during startup...")
    try:
        nltk.download('omw-1.4', download_dir=nltk_data_path, quiet=True)
        print("NLTK 'omw-1.4' downloaded.")
    except Exception as e:
        print(f"Error downloading 'omw-1.4' during startup: {e}")

# Download the vader_lexicon for TextBlob's sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon')
    print("NLTK 'vader_lexicon' found.")
except LookupError:
    print("NLTK 'vader_lexicon' not found. Attempting download during startup...")
    try:
        nltk.download('vader_lexicon', download_dir=nltk_data_path, quiet=True)
        print("NLTK 'vader_lexicon' downloaded.")
    except Exception as e:
        print(f"Error downloading 'vader_lexicon' during startup: {e}")

print("NLTK resource check complete for API script.")

stop_words = set(stopwords.words('english'))
sentiment_words_to_keep = {'love', 'hate', 'like', 'dislike', 'good', 'bad', 'not', 'very', 'too', 'less', 'more', 'no',
                           'healthy', 'unhealthy', 'sick', 'well', 'pain', 'food', 'eat', 'drink', 'exercise', 'stress',
                           'never', 'none', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'aint', 'isnt', 'arent',
                           'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wont', 'wouldnt', 'dont', 'doesnt', 'didnt',
                           'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'}
stop_words = stop_words - sentiment_words_to_keep

# --- Lemmatizer Initialization ---
lemmatizer = WordNetLemmatizer()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable not set!")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
def preprocess_text(text):
    """
    Cleans and preprocesses text: lowercases, removes URLs, mentions, hashtags, punctuation,
    tokenizes, handles simple negation, removes custom stopwords, and lemmatizes.
    This function MUST be identical to the one used during training.
    Handles potential non-string inputs gracefully.
    """
    if not isinstance(text, str): # Ensure input is string
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    word_tokens = word_tokenize(text)

    negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely',
                      'aint', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wont',
                      'wouldnt', 'dont', 'doesnt', 'didnt', 'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'}
    processed_tokens = []
    i = 0
    while i < len(word_tokens):
        word = word_tokens[i]
        if word in negation_words and i + 1 < len(word_tokens):
            processed_tokens.append(word + '_' + word_tokens[i+1])
            i += 2
        else:
            processed_tokens.append(word)
            i += 1

    filtered_words = [
        lemmatizer.lemmatize(word) for word in processed_tokens if word not in stop_words and word.replace('_', '').isalpha()
    ]
    return " ".join(filtered_words)
def analyze_sentiment(text):
    """Analyzes sentiment using TextBlob."""
    if not isinstance(text, str) or not text.strip():
        return 0.0 # Return neutral sentiment for empty or non-string text
    try:
        return TextBlob(text).sentiment.polarity
    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return 0.0 # Return neutral sentiment on error
unhealthy_keywords = [
    "pizza", "burger", "fries", "soda", "sugar", "candy", "chocolate",
    "smoking", "alcohol", "junk food", "fast food", "sedentary", "couch potato",
    "stay up late", "all night", "excessive screen time", "unhealthy food", "cigarettes",
    "binge eating", "crash diet", "skipping meals", "too much screen time", "lack of sleep"
]
strong_unhealthy_keywords = [
    "smoking", "alcohol", "cigarettes", "binge eating", "crash diet", "skipping meals",
    "excessive screen time", "lack of sleep", "stay up all night"
]
avoidance_keywords = [
    "quit", "stopped", "avoid", "hate", "dislike", "giving up", "staying away from",
    "don't eat", "don't drink", "not eating", "not drinking", "avoiding"
]
warning_keywords = [
    "bad for your health", "harmful", "dangerous", "risk", "causes", "leads to",
    "beware of", "warning", "should avoid", "negative effects", "detrimental"
]

known_healthy_habits = [
    "working out in the morning", "exercise", "eating vegetables",
    "drinking water", "sleeping well", "meditation", "yoga", "healthy food",
    "running", "swimming", "gym", "workout", "salad", "fruit", "vegetables",
    "balanced diet", "staying hydrated", "getting enough sleep", "mindfulness", "walking",
    "clean eating", "portion control"
]

def contains_keyword(text, keywords):
    """Checks if the text contains any of the specified keywords."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            return True
    return False
def generate_persuasive_message(unhealthy_habit_text):
    """Generates a persuasive message to encourage a healthier choice."""
    # Check if the API key is available before making the request
    if not OPENROUTER_API_KEY:
        print("Skipping persuasive message generation: OPENROUTER_API_KEY not set.")
        return "Could not generate a persuasive message (API key not set)."
    print("\n--- Generating Persuasive Message ---") # Debug Print
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"Reply to this tweet persuasively, encouraging a healthier choice: '{unhealthy_habit_text}'"
    data = {
        "model": "mistralai/mixtral-8x7b-instruct", # You can experiment with other models
        "messages": [
            {"role": "system", "content": "You are a helpful and encouraging health assistant. Respond like a tweet reply—concise, persuasive, and engaging. Focus on motivating healthier choices."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 75,
        "temperature": 0.8
    }
    message = "Could not generate a persuasive message." # Default message if API call fails
    try:
        response = requests.post(OPENROUTER_URL, json=data, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        message = response_json.get("choices", [{}])[0].get("message", {}).get("content", message)
        print("🗣️ Persuasive Message Generated.") # Debug Print
    except requests.exceptions.RequestException as e:
        print(f"❌ Error generating persuasive message: {e}")
        print("Please check your OpenRouter API key, network connection, and the API endpoint.")
        # Keep the default message or provide a specific error message
        message = f"Error generating message: {e}"[:100] + "..." # Truncate error message for response
    except Exception as e:
        print(f"❌ An unexpected error occurred during message generation: {e}")
        message = f"Unexpected error generating message: {e}"[:100] + "..."
    return message

MODEL_PATH = "best_cnn_lstm_model_deeper.keras"
TOKENIZER_PATH = "tokenizer_config.json"
MAX_LEN = 100
print(f"Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please ensure the model file exists in the same directory.")

    exit()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print(f"Loading tokenizer from {TOKENIZER_PATH}...")
if not os.path.exists(TOKENIZER_PATH):
    print(f"Error: Tokenizer file not found at {TOKENIZER_PATH}")
    print("Please ensure 'tokenizer_config.json' exists in the same directory.")
    exit()
try:
    with open(TOKENIZER_PATH, 'r', encoding='utf-8') as f:
        tokenizer_json_string = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json_string)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()
model_high_confidence_threshold = 0.8
negative_sentiment_threshold = -0.1
positive_sentiment_threshold = 0.1
model_default_threshold = 0.5
strong_override_sentiment_threshold = 0.5
negative_sentiment_unhealthy_avoidance_override_threshold = -0.3
app = FastAPI()

class PostRequest(BaseModel):
    text: str

@app.post("/classify_post")
def classify(post: PostRequest):
    """
    Receives text input, preprocesses it, classifies using the Keras model,
    applies post-processing rules based on sentiment and keywords,
    and returns the final classification, model confidence, and potentially
    a persuasive message if classified as unhealthy.
    """
    original_text = post.text
    print(f"\n--- Received Text: '{original_text}' ---") # Debug Print

    # Handle empty input text early
    if not original_text or not original_text.strip():
        return {"category": "Could not classify", "confidence": 0.0, "detail": "Input text is empty."}
    processed_text = preprocess_text(original_text)
    print(f"Processed Text for Model: '{processed_text}'")
    if not processed_text.strip():
        print("Warning: Preprocessing resulted in empty text.")
        sentiment_polarity = analyze_sentiment(original_text)
        contains_any_unhealthy_kwd = contains_keyword(original_text, unhealthy_keywords)
        contains_strong_unhealthy_kwd = contains_keyword(original_text, strong_unhealthy_keywords)
        contains_any_healthy_kwd = contains_keyword(original_text, known_healthy_habits)
        contains_avoidance_kwd = contains_keyword(original_text, avoidance_keywords)
        contains_warning_kwd = contains_keyword(original_text, warning_keywords)

        final_classification = "Could not classify (Empty after preprocessing)"
        detail = "Preprocessing resulted in empty text. Attempted keyword fallback."
        persuasive_message = "No message generated due to preprocessing issue."
        model_confidence = 0.0 # Confidence is 0 as model was not used
        if contains_strong_unhealthy_kwd:
            final_classification = "unhealthy"
            detail = "Classified as UNHEALTHY based on strong keyword in original text (empty after preprocessing)."
            persuasive_message = generate_persuasive_message(original_text)
        elif contains_any_unhealthy_kwd and not contains_any_healthy_kwd:
            final_classification = "unhealthy"
            detail = "Classified as UNHEALTHY based on any unhealthy keyword in original text (empty after preprocessing)."
            persuasive_message = generate_persuasive_message(original_text)
        elif contains_any_healthy_kwd and not contains_any_unhealthy_kwd:
            final_classification = "healthy"
            detail = "Classified as HEALTHY based on healthy keyword in original text (empty after preprocessing)."
            persuasive_message = "No message generated (classified as healthy)." # No message for healthy fallback
        else:
            final_classification = "Could not confidently classify"
            detail = "Preprocessing resulted in empty text. Keyword fallback inconclusive."
            persuasive_message = "No message generated (could not classify)."
        print(f"Fallback Classification: {final_classification}, Detail: {detail}")
        return {
            "category": final_classification.capitalize(), # Capitalize for nice output
            "confidence": model_confidence,
            "detail": detail,
            "persuasive_message": persuasive_message
        }
    sequences = tokenizer.texts_to_sequences([processed_text])
    if not sequences or not sequences[0]:
        print("Warning: Tokenizer returned empty sequence after preprocessing.")
        # Fallback logic similar to empty processed text, but after tokenization
        sentiment_polarity = analyze_sentiment(original_text)
        contains_any_unhealthy_kwd = contains_keyword(original_text, unhealthy_keywords)
        contains_strong_unhealthy_kwd = contains_keyword(original_text, strong_unhealthy_keywords)
        contains_any_healthy_kwd = contains_keyword(original_text, known_healthy_habits)
        contains_avoidance_kwd = contains_keyword(original_text, avoidance_keywords)
        contains_warning_kwd = contains_keyword(original_text, warning_keywords)

        final_classification = "Could not classify (Empty after tokenization)"
        detail = "Tokenizer returned empty sequence. Attempted keyword fallback."
        persuasive_message = "No message generated due to tokenization issue."
        model_confidence = 0.0 # Confidence is 0 as model was not used
        if contains_strong_unhealthy_kwd:
            final_classification = "unhealthy"
            detail = "Classified as UNHEALTHY based on strong keyword in original text (empty after tokenization)."
            persuasive_message = generate_persuasive_message(original_text)
        elif contains_any_unhealthy_kwd and not contains_any_healthy_kwd:
            final_classification = "unhealthy"
            detail = "Classified as UNHEALTHY based on any unhealthy keyword in original text (empty after tokenization)."
            persuasive_message = generate_persuasive_message(original_text)
        elif contains_any_healthy_kwd and not contains_any_unhealthy_kwd:
            final_classification = "healthy"
            detail = "Classified as HEALTHY based on healthy keyword in original text (empty after tokenization)."
            persuasive_message = "No message generated (classified as healthy)."
        else:
            final_classification = "Could not confidently classify"
            detail = "Tokenizer returned empty sequence. Keyword fallback inconclusive."
            persuasive_message = "No message generated (could not classify)."

        print(f"Fallback Classification: {final_classification}, Detail: {detail}")

        return {
            "category": final_classification.capitalize(),
            "confidence": model_confidence,
            "detail": detail,
            "persuasive_message": persuasive_message
        }
    padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post', dtype='float32')

    try:
        predictions = model.predict(padded, verbose=0)
        # Assuming a single output neuron with sigmoid activation
        probability_unhealthy = predictions[0][0]
        probability_healthy = 1 - probability_unhealthy
        model_confidence = float(probability_unhealthy) # Use unhealthy probability as confidence

        print(f"\n--- Model Prediction (Raw) ---") # Debug Print
        print(f"🔹 Probability (Unhealthy): {probability_unhealthy:.4f}")
        print(f"🔹 Probability (Healthy): {probability_healthy:.4f}")
        print("------------------------------")

    except Exception as e:
        print(f"❌ Error during model prediction: {e}")
        sentiment_polarity = analyze_sentiment(original_text)
        contains_any_unhealthy_kwd = contains_keyword(original_text, unhealthy_keywords)
        contains_strong_unhealthy_kwd = contains_keyword(original_text, strong_unhealthy_keywords)
        contains_any_healthy_kwd = contains_keyword(original_text, known_healthy_habits)
        contains_avoidance_kwd = contains_keyword(original_text, avoidance_keywords)
        contains_warning_kwd = contains_keyword(original_text, warning_keywords)

        final_classification = "Could not classify (Model Error)"
        detail = f"Error during model prediction: {e}"[:100] + "..." + ". Attempted keyword fallback."
        persuasive_message = "No message generated due to model error."
        model_confidence = 0.0 # Confidence is 0 as model failed

        if contains_strong_unhealthy_kwd:
            final_classification = "unhealthy"
            detail = "Classified as UNHEALTHY based on strong keyword (Model Error Fallback)."
            persuasive_message = generate_persuasive_message(original_text)
        elif contains_any_unhealthy_kwd and not contains_any_healthy_kwd:
            final_classification = "unhealthy"
            detail = "Classified as UNHEALTHY based on any unhealthy keyword (Model Error Fallback)."
            persuasive_message = generate_persuasive_message(original_text)
        elif contains_any_healthy_kwd and not contains_any_unhealthy_kwd:
            final_classification = "healthy"
            detail = "Classified as HEALTHY based on healthy keyword (Model Error Fallback)."
            persuasive_message = "No message generated (classified as healthy)."
        else:
            final_classification = "Could not confidently classify"
            detail = "Model Error Fallback: Keyword fallback inconclusive."
            persuasive_message = "No message generated (could not classify)."

        print(f"Fallback Classification: {final_classification}, Detail: {detail}")

        return {
            "category": final_classification.capitalize(),
            "confidence": model_confidence,
            "detail": detail,
            "persuasive_message": persuasive_message
        }

    try:
        sentiment_polarity = analyze_sentiment(original_text)
        contains_any_unhealthy_kwd = contains_keyword(original_text, unhealthy_keywords)
        contains_strong_unhealthy_kwd = contains_keyword(original_text, strong_unhealthy_keywords)
        contains_any_healthy_kwd = contains_keyword(original_text, known_healthy_habits)
        contains_avoidance_kwd = contains_keyword(original_text, avoidance_keywords)
        contains_warning_kwd = contains_keyword(original_text, warning_keywords)

        print(f"--- Post-processing Info ---") # Debug Print
        print(f"🔹 Sentiment Polarity: {sentiment_polarity:.4f}")
        print(f"🔹 Contains Any Unhealthy Keyword: {contains_any_unhealthy_kwd}")
        print(f"🔹 Contains Strong Unhealthy Keyword: {contains_strong_unhealthy_kwd}")
        print(f"🔹 Contains Any Healthy Keyword: {contains_any_healthy_kwd}")
        print(f"🔹 Contains Avoidance Keyword: {contains_avoidance_kwd}")
        print(f"🔹 Contains Warning Keyword: {contains_warning_kwd}")
        print("--------------------------")

        print(f"--- Debugging Post-processing Conditions ---") # Debug Print
        print(f"Thresholds: High Conf={model_high_confidence_threshold}, Neg Sent={negative_sentiment_threshold}, Pos Sent={positive_sentiment_threshold}, Default Model={model_default_threshold}, Strong Override Sent={strong_override_sentiment_threshold}, Neg Sent Unhealthy Avoidance Override={negative_sentiment_unhealthy_avoidance_override_threshold}")
        print(f"Model Prob Unhealthy: {probability_unhealthy:.4f}, Prob Healthy: {probability_healthy:.4f}")
        print(f"Sentiment Polarity: {sentiment_polarity:.4f}")
        print(f"Contains Any Unhealthy Kwd: {contains_any_unhealthy_kwd}, Strong Unhealthy Kwd: {contains_strong_unhealthy_kwd}, Any Healthy Kwd: {contains_any_healthy_kwd}, Avoidance Kwd: {contains_avoidance_kwd}, Warning Kwd: {contains_warning_kwd}")
        print("--------------------------------------------")
        final_classification = "healthy" # Default initial classification
        classification_detail = "Model default" # Detail for debugging

        if contains_any_unhealthy_kwd and contains_warning_kwd and sentiment_polarity < negative_sentiment_threshold:
            final_classification = "healthy"
            classification_detail = "Rule 0: Unhealthy + Warning + Negative Sentiment (Awareness)"
            print("✅ Post-processing Rule 0 Applied: Classified as HEALTHY (Awareness Statement).") # Debug Print

        elif contains_strong_unhealthy_kwd:
            if (sentiment_polarity > strong_override_sentiment_threshold and
                    contains_avoidance_kwd and
                    not contains_any_healthy_kwd):
                final_classification = "healthy"
                classification_detail = "Rule 1 - Override Strong Unhealthy: Strong unhealthy, BUT VERY positive sentiment + Avoidance (no healthy)"
                print("✅ Post-processing override (Rule 1 - Override Strong Unhealthy) Applied: Classified as HEALTHY.") # Debug Print
            else:
                # Otherwise, classify as UNHEALTHY
                final_classification = "unhealthy"
                classification_detail = "Rule 1: Strong unhealthy keyword"
                print("🚨 Post-processing Rule 1 Applied: Classified as UNHEALTHY.") # Debug Print

        elif contains_any_unhealthy_kwd:
            if (sentiment_polarity < negative_sentiment_unhealthy_avoidance_override_threshold and
                    contains_avoidance_kwd):
                final_classification = "healthy"
                classification_detail = "Rule 2a - Negative Sentiment + Avoidance on Unhealthy"
                print("✅ Post-processing override (Rule 2a - Negative Sentiment + Avoidance on Unhealthy) Applied: Classified as HEALTHY.") # Debug Print
            elif probability_unhealthy > model_high_confidence_threshold:
                final_classification = "unhealthy"
                classification_detail = "Rule 2b: Any unhealthy + High unhealthy probability"
                print("🚨 Post-processing Rule 2b Applied: Classified as UNHEALTHY.") # Debug Print
            elif probability_healthy > model_high_confidence_threshold and sentiment_polarity < negative_sentiment_threshold:
                final_classification = "unhealthy"
                classification_detail = "Rule 2c - Override Any Unhealthy: Any unhealthy + High healthy prob + Negative sentiment"
                print("🚨 Post-processing override (Rule 2c - Override Any Unhealthy) Applied: Classified as UNHEALTHY.") # Debug Print
            else:
                if probability_unhealthy > model_default_threshold:
                    final_classification = "unhealthy"
                    classification_detail = "Rule 2d Default: Any unhealthy, Model UNHEALTHY"
                    print("🚨 Model Classified (Rule 2d Default) Applied: Classified as UNHEALTHY.") # Debug Print
                else:
                    final_classification = "healthy"
                    classification_detail = "Rule 2d Default: Any unhealthy, Model HEALTHY"
                    print("✅ Model Classified (Rule 2d Default) Applied: Classified as HEALTHY.") # Debug Print

        elif contains_any_healthy_kwd:
            if probability_healthy > model_high_confidence_threshold:
                final_classification = "healthy"
                classification_detail = "Rule 3a: Any healthy + High healthy probability"
                print("✅ Post-processing Rule 3a Applied: Classified as HEALTHY.") # Debug Print
            elif probability_unhealthy > model_high_confidence_threshold and sentiment_polarity > positive_sentiment_threshold:
                final_classification = "healthy"
                classification_detail = "Rule 3b - Override Any Healthy: Any healthy + High unhealthy prob + Positive sentiment"
                print("✅ Post-processing override (Rule 3b - Override Any Healthy) Applied: Classified as HEALTHY.") # Debug Print
            else:
                if probability_unhealthy > model_default_threshold:
                    final_classification = "unhealthy"
                    classification_detail = "Rule 3c Default: Any healthy, Model UNHEALTHY"
                    print("🚨 Model Classified (Rule 3c Default) Applied: Classified as UNHEALTHY.") # Debug Print
                else:
                    final_classification = "healthy"
                    classification_detail = "Rule 3c Default: Any healthy, Model HEALTHY"
                    print("✅ Model Classified (Rule 3c Default) Applied: Classified as HEALTHY.") # Debug Print

        else:
            if probability_unhealthy > model_default_threshold:
                final_classification = "unhealthy"
                classification_detail = "Rule 4 - Default: No keyword, Model UNHEALTHY"
                print("🚨 Model Classified (Rule 4 - Default) Applied: Classified as UNHEALTHY.") # Debug Print
            else:
                final_classification = "healthy"
                classification_detail = "Rule 4 - Default: No keyword, Model HEALTHY"
                print("✅ Model Classified (Rule 4 - Default) Applied: Classified as HEALTHY.") # Debug Print

        print(f"--- Final Classification: {final_classification} ---") # Debug Print

    except Exception as e:
        print(f"❌ Error during post-processing logic: {e}")

        final_classification = "Could not finalize classification (Post-processing Error)"
        detail = f"Error during post-processing logic: {e}"[:100] + "..."
        persuasive_message = "No message generated due to processing error."

        fallback_category_from_model = "unhealthy" if model_confidence > model_default_threshold else "healthy"
        final_classification = f"{fallback_category_from_model.capitalize()} (Post-processing Error Fallback)"
        detail = f"Post-processing Error: {e}"[:100] + "... Falling back to model prediction."

        if fallback_category_from_model == "unhealthy":
            persuasive_message = generate_persuasive_message(original_text)
        else:
            persuasive_message = "No message generated (classified as healthy by fallback)."
        print(f"Fallback Classification: {final_classification}, Detail: {detail}")
        return {
            "category": final_classification,
            "confidence": model_confidence,
            "detail": detail,
            "persuasive_message": persuasive_message
        }
    persuasive_message = "No message generated." # Default message output
    if final_classification == "unhealthy":
        persuasive_message = generate_persuasive_message(original_text)
    else:
        persuasive_message = "No message generated (classified as healthy)."

    return {
        "category": final_classification.capitalize(), # Capitalize for nice output
        "confidence": model_confidence,
        "detail": classification_detail,
        "persuasive_message": persuasive_message
    }
