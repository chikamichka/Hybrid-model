# main.py in your functions directory

import functions_framework
import tensorflow as tf
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # Import Lemmatizer
from textblob import TextBlob  # Import TextBlob for sentiment analysis
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
import requests # Import requests for OpenRouter API calls

# --- NLTK Downloads (Needed for the function environment) ---
# IMPORTANT FOR PRODUCTION: Downloading NLTK data at runtime can be slow and unreliable.
# It is highly recommended to bundle the necessary NLTK data with your function deployment
# instead of relying on these downloads.
# See: https://firebase.google.com/docs/functions/callable-functions#include_sdk
# The code below attempts to download if not found, but bundling is preferred.
import nltk
print("Checking for NLTK resources for function...")
try:
    nltk.data.find('corpora/stopwords')
    print("NLTK 'stopwords' found.")
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'stopwords' not found. Attempting download...")
    try:
        nltk.download('stopwords')
        print("NLTK 'stopwords' downloaded.")
    except Exception as e:
        print(f"Error downloading NLTK 'stopwords': {e}")


try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' found.")
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'punkt' not found. Attempting download...")
    try:
        nltk.download('punkt')
        print("NLTK 'punkt' downloaded.")
    except Exception as e:
        print(f"Error downloading NLTK 'punkt': {e}")

try:
    nltk.data.find('corpora/wordnet') # Check for WordNet
    print("NLTK 'wordnet' found.")
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'wordnet' not found. Attempting download...")
    try:
        nltk.download('wordnet')
        print("NLTK 'wordnet' downloaded.")
    except Exception as e:
        print(f"Error downloading NLTK 'wordnet': {e}")

try:
    nltk.data.find('corpora/omw-1.4') # Check for Open Multilingual Wordnet
    print("NLTK 'omw-1.4' found.")
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'omw-1.4' not found. Attempting download...")
    try:
        nltk.download('omw-1.4')
        print("NLTK 'omw-1.4' downloaded.")
    except Exception as e:
        print(f"Error downloading NLTK 'omw-1.4': {e}")


# Download the vader_lexicon for TextBlob's sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon')
    print("NLTK 'vader_lexicon' found.")
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'vader_lexicon' not found. Attempting download...")
    try:
        nltk.download('vader_lexicon')
        print("NLTK 'vader_lexicon' downloaded.")
    except Exception as e:
        print(f"Error downloading NLTK 'vader_lexicon': {e}")

print("NLTK resource check complete for function.")


# --- Custom Stopword Set (MUST match train_model.py) ---
stop_words = set(stopwords.words('english'))

# Define words that should NOT be removed, even if they are in the standard stopword list
# MUST match the set used in train_model.py
sentiment_words_to_keep = {'love', 'hate', 'like', 'dislike', 'good', 'bad', 'not', 'very', 'too', 'less', 'more', 'no',
                           'healthy', 'unhealthy', 'sick', 'well', 'pain', 'food', 'eat', 'drink', 'exercise', 'stress',
                           'never', 'none', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'aint', 'isnt', 'arent',
                           'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wont', 'wouldnt', 'dont', 'doesnt', 'didnt',
                           'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'}

# Remove these sentiment words from the stopword set
stop_words = stop_words - sentiment_words_to_keep

# --- Lemmatizer Initialization ---
lemmatizer = WordNetLemmatizer()


# --- Preprocessing function (MUST match train_model.py's preprocess_text_enhanced) ---
def preprocess_text(text):
    """
    Cleans and preprocesses text: lowercases, removes URLs, mentions, hashtags, punctuation,
    tokenizes, handles simple negation, removes custom stopwords, and lemmatizes.
    This function MUST be identical to the one used during training.
    """
    if not isinstance(text, str): # Ensure input is string
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs more robustly
    text = re.sub(r'@\w+', '', text) # Remove mentions (@user)
    text = re.sub(r'#', '', text) # Remove hashtags (#) - keeping the word, removing the symbol
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation

    word_tokens = word_tokenize(text)

    # Simple Negation Handling: Join negation words with the next word
    # MUST match the list used in train_model.py
    negation_words = {'not', 'no', 'never', 'none', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely',
                      'aint', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wont',
                      'wouldnt', 'dont', 'doesnt', 'didnt', 'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'}
    processed_tokens = []
    i = 0
    while i < len(word_tokens):
        word = word_tokens[i]
        # Check if the current word is a negation and there's a next word
        if word in negation_words and i + 1 < len(word_tokens):
            # Join negation word with the next word using an underscore
            processed_tokens.append(word + '_' + word_tokens[i+1])
            i += 2 # Skip the next word as it's now part of the negation token
        else:
            # If not a negation or no next word, just append the current word
            processed_tokens.append(word)
            i += 1

    # Lemmatize and remove stop words using the custom set
    # Apply lemmatization *after* negation handling
    # Check if the word (or joined word) consists only of alphabetic characters (ignoring the underscore for joined words)
    filtered_words = [
        lemmatizer.lemmatize(word) for word in processed_tokens if word not in stop_words and word.replace('_', '').isalpha()
    ]
    return " ".join(filtered_words)


# --- Load Global Resources (Model, Tokenizer) ---
# Load these outside the function definition so they are only loaded once
# when the function instance starts, not on every invocation.

tokenizer = None
# Assuming tokenizer_config.json is in the same directory as main.py
tokenizer_config_path = os.path.join(os.path.dirname(__file__), 'tokenizer_config.json')
try:
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        tokenizer_json_string = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json_string)
    print("Global: Tokenizer loaded successfully.")
except Exception as e:
    print(f"Global Error loading tokenizer configuration: {e}")
    # In a real function, you might want to log this error and handle function startup failure


interpreter = None
# Use the path to your LATEST saved TFLite model (optimized recommended)
# Make sure this path matches the output filename from your last train_model.py run
model_filename = 'best_cnn_lstm_model_optimized_deeper.tflite' # <--- !! UPDATED FILENAME !!
model_path = os.path.join(os.path.dirname(__file__), model_filename)
try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print("Global: TFLite model loaded and tensors allocated successfully.")
except Exception as e:
    print(f"Global Error loading TFLite model: {e}")
    # Log error, handle function startup failure

max_len = None
input_dtype = None
if interpreter:
    try:
        input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details() # Not strictly needed globally
        max_len = input_details[0]['shape'][1]
        input_dtype = input_details[0]['dtype']
        print(f"Global: Detected max_len: {max_len}, Input dtype: {input_dtype}")
    except Exception as e:
        print(f"Global Error getting model details: {e}")
        # Log error, handle function startup failure


# --- OpenRouter API Details ---
# NOTE: Hardcoding API keys like this is NOT secure for production apps.
# For a real Flutter app, the API call should be made from a secure backend.
# Use Firebase Environment Configuration for API keys:
# https://firebase.google.com/docs/functions/config-env
# Example using functions.config (requires firebase-functions SDK):
# import firebase_functions.params as params
# OPENROUTER_API_KEY = params.Secret('OPENROUTER_API_KEY').value()
# Or using os.environ with functions-framework:
# OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
# For this code, keeping the hardcoded key for demonstration, BUT CHANGE THIS!
OPENROUTER_API_KEY = "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" # REPLACE WITH SECURE METHOD


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Helper: Sentiment analysis (using TextBlob) ---
def analyze_sentiment(text):
    """Analyzes sentiment using TextBlob."""
    # TextBlob returns a Sentiment object with polarity and subjectivity
    # Ensure vader_lexicon is available (handled by global NLTK download attempts)
    try:
        return TextBlob(text).sentiment.polarity
    except LookupError:
        print("Sentiment analysis failed: NLTK 'vader_lexicon' not found.")
        return 0.0 # Return neutral sentiment if lexicon is missing


# --- Define keywords associated with unhealthy items/behaviors ---
unhealthy_keywords = [
    "pizza", "burger", "fries", "soda", "sugar", "candy", "chocolate",
    "smoking", "alcohol", "junk food", "fast food", "sedentary", "couch potato",
    "stay up late", "all night", "excessive screen time", "unhealthy food", "cigarettes",
    "binge eating", "crash diet", "skipping meals", "too much screen time", "lack of sleep"
    # Add more as needed - MUST be consistent with keywords considered in training data/logic
]

# --- Define highly indicative unhealthy keywords ---
strong_unhealthy_keywords = [
    "smoking", "alcohol", "cigarettes", "binge eating", "crash diet", "skipping meals",
    "excessive screen time", "lack of sleep", "stay up all night"
    # Add more keywords that are almost always unhealthy habits
]

# --- Define keywords indicating avoidance or quitting ---
avoidance_keywords = [
    "quit", "stopped", "avoid", "hate", "dislike", "giving up", "staying away from",
    "don't eat", "don't drink", "not eating", "not drinking", "avoiding"
    # Add more phrases indicating avoidance or quitting
]

# --- Define keywords indicating a warning or awareness statement ---
warning_keywords = [
    "bad for your health", "harmful", "dangerous", "risk", "causes", "leads to",
    "beware of", "warning", "should avoid", "negative effects", "detrimental"
    # Add more phrases indicating a warning or negative consequence
]


def contains_keyword(text, keywords):
    """Checks if the text contains any of the specified keywords."""
    if not isinstance(text, str):
        return False
    text_lower = text.lower()
    # Check for exact words or phrases in the list
    for keyword in keywords:
        # Use word boundaries to avoid matching parts of words (e.g., "sugar" in "sugary")
        if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
            return True
    return False


# Known healthy habits (from your previous script) - Can be used for quick checks
known_healthy_habits = [
    "working out in the morning", "exercise", "eating vegetables",
    "drinking water", "sleeping well", "meditation", "yoga", "healthy food",
    "running", "swimming", "gym", "workout", "salad", "fruit", "vegetables",
    "balanced diet", "staying hydrated", "getting enough sleep", "mindfulness", "walking",
    "clean eating", "portion control"
    # Add more well-known healthy phrases - MUST be consistent
]

# --- Function to generate persuasive message using OpenRouter ---
def generate_persuasive_message(unhealthy_habit_text):
    """Generates a persuasive message to encourage a healthier choice."""
    print("\n--- Generating Persuasive Message ---")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional: Specify your site URL and GitHub repo if you have one
        # "HTTP-Referer": "YOUR_SITE_URL", # Recommended for OpenRouter
        # "X-Title": "YOUR_APP_NAME" # Recommended for OpenRouter
    }

    # Craft a prompt that encourages a helpful and persuasive response
    prompt = f"Reply to this tweet persuasively, encouraging a healthier choice: '{unhealthy_habit_text}'"

    data = {
        "model": "mistralai/mixtral-8x7b-instruct", # You can experiment with other models
        "messages": [
            {"role": "system", "content": "You are a helpful and encouraging health assistant. Respond like a tweet reply—concise, persuasive, and engaging. Focus on motivating healthier choices."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 75, # Increased max_tokens slightly for potentially better messages
        "temperature": 0.8 # Adjusted temperature for potentially more creative/persuasive text
    }

    try:
        response = requests.post(OPENROUTER_URL, json=data, headers=headers)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        response_json = response.json()
        message = response_json.get("choices", [{}])[0].get("message", {}).get("content", "Could not generate a persuasive message.")
        print("Persuasive Message generated successfully.")
        return message
    except requests.exceptions.RequestException as e:
        print(f"❌ Error generating persuasive message: {e}")
        print("Please check your OpenRouter API key, network connection, and the API endpoint.")
        return "Error generating message."
    except Exception as e:
        print(f"❌ An unexpected error occurred during message generation: {e}")
        return "Error generating message."
    finally:
        print("-------------------------------------")


# --- Classifier Logic for TFLite Model with Post-processing ---
def classify_text_logic(original_text):
    """
    Contains the core classification and post-processing logic.
    Separated from the HTTP trigger for clarity and potential reuse.
    """
    # Ensure global resources are available
    if interpreter is None or tokenizer is None or max_len is None or input_dtype is None:
        print("Error: Global model resources not loaded.")
        return {"classification": "error", "message": "Backend model resources not ready."}

    # --- Run Model Inference ---
    processed_text = preprocess_text(original_text)
    print(f"Processed Text for Model: '{processed_text}'")

    sequences = tokenizer.texts_to_sequences([processed_text])
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post', dtype=input_dtype.__name__)
    input_data = np.array(padded_sequences, dtype=input_dtype)

    expected_input_shape = interpreter.get_input_details()[0]['shape']

    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=0)

    if list(input_data.shape) != list(expected_input_shape):
        print(f"Error: Input data shape {input_data.shape} does not match expected model input shape {expected_input_shape}.")
        return {"classification": "error", "message": "Input shape mismatch."}

    try:
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    except Exception as e:
        print(f"Error during model inference: {e}")
        return {"classification": "error", "message": f"Error during model inference: {e}"}

    probability_unhealthy = None
    if output_data.shape == (1, 1):
        probability_unhealthy = output_data[0][0]
    else:
        print(f"Warning: Unexpected model output shape: {output_data.shape}")
        # Attempt to handle potential different output shapes if they are probabilities
        if interpreter.get_output_details()[0]['shape'][-1] == 1:
            probability_unhealthy = output_data.flatten()[0]
            print(f"Attempted to flatten output and got probability: {probability_unhealthy:.4f}")
        else:
            print("Could not extract single probability from output.")
            return {"classification": "error", "message": "Unexpected model output shape."}


    probability_healthy = 1 - probability_unhealthy

    print(f"Model Prediction: Unhealthy={probability_unhealthy:.4f}, Healthy={probability_healthy:.4f}")

    # --- Post-processing with Sentiment and Keywords ---
    sentiment_polarity = analyze_sentiment(original_text)
    contains_any_unhealthy_kwd = contains_keyword(original_text, unhealthy_keywords)
    contains_strong_unhealthy_kwd = contains_keyword(original_text, strong_unhealthy_keywords)
    contains_any_healthy_kwd = contains_keyword(original_text, known_healthy_habits)
    contains_avoidance_kwd = contains_keyword(original_text, avoidance_keywords)
    contains_warning_kwd = contains_keyword(original_text, warning_keywords)


    print(f"Post-processing Info: Sentiment={sentiment_polarity:.4f}, Any Unhealthy={contains_any_unhealthy_kwd}, Strong Unhealthy={contains_strong_unhealthy_kwd}, Any Healthy={contains_any_healthy_kwd}, Avoidance={contains_avoidance_kwd}, Warning={contains_warning_kwd}")


    # Define thresholds for post-processing
    model_high_confidence_threshold = 0.8
    negative_sentiment_threshold = -0.1
    positive_sentiment_threshold = 0.1
    model_default_threshold = 0.5
    strong_override_sentiment_threshold = 0.5
    negative_sentiment_unhealthy_avoidance_override_threshold = -0.3


    final_classification = "healthy" # Default initial classification


    # --- Post-processing Logic (Matching testo.py v6) ---
    # Prioritize warning statements.
    # Then give higher priority to strong unhealthy keywords.
    # Handle negative sentiment towards unhealthy items ONLY when avoidance is indicated.

    # Rule 0: If it's a warning/awareness statement about an unhealthy habit
    if contains_any_unhealthy_kwd and contains_warning_kwd and sentiment_polarity < negative_sentiment_threshold:
        final_classification = "healthy"
        print("✅ Post-processing Rule 0: Unhealthy keyword + Warning keyword + Negative sentiment -> Classified as **HEALTHY** (Awareness Statement).")

    # Rule 1: If a STRONG unhealthy keyword is present (and not a warning statement, handled by Rule 0)
    elif contains_strong_unhealthy_kwd:
        # Override to HEALTHY only if sentiment is VERY strongly positive AND avoidance keyword is present AND no healthy keyword is present
        if (sentiment_polarity > strong_override_sentiment_threshold and
                contains_avoidance_kwd and
                not contains_any_healthy_kwd):
            final_classification = "healthy"
            print("✅ Post-processing override (Rule 1 - Override Strong Unhealthy): Strong unhealthy keyword, BUT VERY strong positive sentiment + Avoidance keyword (no healthy keyword) -> Classified as **HEALTHY**.")
        else:
            # Otherwise, classify as UNHEALTHY
            final_classification = "unhealthy"
            print("🚨 Post-processing Rule 1: Strong unhealthy keyword present -> Classified as **UNHEALTHY**.")

    # Rule 2: If ANY unhealthy keyword is present (but not a strong one or a warning, handled by Rule 0/1)
    elif contains_any_unhealthy_kwd:
        # Sub-Rule 2a: If sentiment is significantly negative AND contains avoidance keywords, classify as HEALTHY
        if (sentiment_polarity < negative_sentiment_unhealthy_avoidance_override_threshold and
                contains_avoidance_kwd):
            final_classification = "healthy"
            print("✅ Post-processing override (Rule 2a - Negative Sentiment + Avoidance on Unhealthy): Any unhealthy keyword + Significantly negative sentiment + Avoidance keyword -> Classified as **HEALTHY**.")
        # If model is highly confident it's unhealthy, confirm UNHEALTHY (unless overridden by 2a)
        elif probability_unhealthy is not None and probability_unhealthy > model_high_confidence_threshold:
            final_classification = "unhealthy"
            print("🚨 Post-processing Rule 2b: Any unhealthy keyword + High unhealthy probability -> Classified as **UNHEALTHY**.")
        # If model is highly confident it's healthy AND sentiment is negative, override to UNHEALTHY (less likely with 2a)
        elif probability_healthy is not None and probability_healthy > model_high_confidence_threshold and sentiment_polarity < negative_sentiment_threshold:
            final_classification = "unhealthy"
            print("🚨 Post-processing override (Rule 2c - Override Any Unhealthy): Any unhealthy keyword + High healthy prob + Negative sentiment -> Classified as **UNHEALTHY**.")
        # Otherwise, rely on the model's default prediction for these cases
        else:
            if probability_unhealthy is not None and probability_unhealthy > model_default_threshold:
                final_classification = "unhealthy"
                print("🚨 Model Classified (Rule 2d Default): Any unhealthy keyword present, Model Classified **UNHEALTHY**.")
            else:
                final_classification = "healthy"
                print("✅ Model Classified (Rule 2d Default): Any unhealthy keyword present, Model Classified **HEALTHY**.")


    # Rule 3: If ANY healthy keyword is present (and no unhealthy keywords or warnings, handled by Rule 0/1/2)
    elif contains_any_healthy_kwd:
        # If model is highly confident it's healthy, confirm HEALTHY
        if probability_healthy is not None and probability_healthy > model_high_confidence_threshold:
            final_classification = "healthy"
            print("✅ Post-processing Rule 3a: Any healthy keyword + High healthy probability -> Classified as **HEALTHY**.")
        # If model is highly confident it's unhealthy AND sentiment is positive, override to HEALTHY
        elif probability_unhealthy is not None and probability_unhealthy > model_high_confidence_threshold and sentiment_polarity > positive_sentiment_threshold:
            final_classification = "healthy"
            print("✅ Post-processing override (Rule 3b - Override Any Healthy): Any healthy keyword + High unhealthy prob + Positive sentiment -> Classified as **HEALTHY**.")
        # Otherwise, rely on the model's default prediction for these cases
        else:
            if probability_unhealthy is not None and probability_unhealthy > model_default_threshold:
                final_classification = "unhealthy"
                print("🚨 Model Classified (Rule 3c Default): Any healthy keyword present, Model Classified **UNHEALTHY**.")
            else:
                final_classification = "healthy"
                print("✅ Model Classified (Rule 3c Default): Any healthy keyword present, Model Classified **HEALTHY**.")

    # Rule 4: If NO relevant keyword is found (and not a warning, handled by Rule 0)
    else:
        # Rely solely on the model's default prediction
        if probability_unhealthy is not None and probability_unhealthy > model_default_threshold:
            final_classification = "unhealthy"
            print("🚨 Model Classified (Rule 4 - Default): No relevant keyword, Model Classified **UNHEALTHY**.")
        else:
            final_classification = "healthy"
            print("✅ Model Classified (Rule 4 - Default): No relevant keyword, Model Classified **HEALTHY**.")


    print(f"Final classification: {final_classification}")

    # --- Prepare Response Data ---
    response_data = {
        "classification": final_classification,
        "probability_unhealthy": float(probability_unhealthy) if probability_unhealthy is not None else None,
        "probability_healthy": float(probability_healthy) if probability_healthy is not None else None,
        "sentiment_polarity": float(sentiment_polarity),
        "contains_any_unhealthy_keyword": bool(contains_any_unhealthy_kwd),
        "contains_strong_unhealthy_keyword": bool(contains_strong_unhealthy_kwd),
        "contains_any_healthy_keyword": bool(contains_any_healthy_kwd),
        "contains_avoidance_keyword": bool(contains_avoidance_kwd),
        "contains_warning_keyword": bool(contains_warning_kwd),
        "persuasive_message": None # Initialize message as None
    }

    # --- Generate persuasive message if classified as UNHEALTHY ---
    if final_classification == "unhealthy":
        persuasive_message = generate_persuasive_message(original_text)
        response_data["persuasive_message"] = persuasive_message # Add message to response data

    return response_data # Return the data dictionary


# --- Cloud Function Triggered by HTTP Request ---
@functions_framework.http
def classify_tweet_http(request):
    """HTTP Cloud Function to classify tweet text and generate a persuasive message if unhealthy."""
    print("--- Function classify_tweet_http triggered ---") # Log function start

    # Set CORS headers for local testing (if needed) and production
    # Allows POST requests from any origin during development
    # In production, restrict this to your app's origin
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    # Ensure model and tokenizer were loaded globally
    # If global loading failed, these will be None
    if interpreter is None or tokenizer is None or max_len is None or input_dtype is None:
        print("Function startup failed: Global model resources not ready.")
        response_data = {"classification": "error", "message": "Backend model resources not ready."}
        return json.dumps(response_data), 500, headers


    # Get the tweet text from the request (assuming JSON body with 'text' key)
    try:
        request_json = request.get_json(silent=True)
        tweet_text = None
        if request_json and 'text' in request_json:
            tweet_text = request_json['text']
        else:
            print("Invalid request: JSON body with 'text' key expected.")
            response_data = {"classification": "error", "message": "Invalid request. Please send JSON body with 'text' key."}
            return json.dumps(response_data), 400, headers

        if not tweet_text or not isinstance(tweet_text, str):
            print("Invalid request: Tweet text is empty or not a string.")
            response_data = {"classification": "error", "message": "Tweet text is empty or invalid."}
            return json.dumps(response_data), 400, headers

    except Exception as e:
        print(f"Error parsing request JSON: {e}")
        response_data = {"classification": "error", "message": f"Error processing request: {e}"}
        return json.dumps(response_data), 400, headers

    # --- Perform Classification and Get Results ---
    classification_results = classify_text_logic(tweet_text)

    # Check if classification logic returned an error
    if classification_results.get("classification") == "error":
        return json.dumps(classification_results), 500, headers # Return error from classification logic

    # --- Return Final Response ---
    # The classification_results dictionary already contains all the necessary data,
    # including the persuasive_message if generated.
    return json.dumps(classification_results), 200, headers
