print("--- VERIFYING: THIS IS THE LATEST TESTO.PY WITH DEBUG AND OPENROUTER ---") # <-- ADD THIS LINE AT THE VERY TOP
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

# --- OpenRouter API details ---
# WARNING: Replace with a secure method for handling API keys in production!
OPENROUTER_API_KEY = "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" # Replace with your actual key
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- NLTK Downloads ---
import nltk
print("Checking for NLTK resources for testing script...")
try:
    stopwords.words('english')
    print("NLTK 'stopwords' found.")
except LookupError:
    print("NLTK 'stopwords' not found. Downloading...")
    nltk.download('stopwords')
    print("NLTK 'stopwords' downloaded.")

try:
    word_tokenize("example")
    print("NLTK 'punkt' found.")
except LookupError:
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download('punkt')
    print("NLTK 'punkt' downloaded.")

try:
    WordNetLemmatizer().lemmatize("example") # Check for WordNet
    print("NLTK 'wordnet' found.")
except LookupError:
    print("NLTK 'wordnet' not found. Downloading...")
    nltk.download('wordnet')
    print("NLTK 'wordnet' downloaded.")

try:
    nltk.data.find('corpora/omw-1.4') # Check for Open Multilingual Wordnet
    print("NLTK 'omw-1.4' found.")
# Catching LookupError directly as it's the error for resource not found
except LookupError:
    print("NLTK 'omw-1.4' not found. Downloading...")
    nltk.download('omw-1.4')
    print("NLTK 'omw-1.4' downloaded.")

# Download the vader_lexicon for TextBlob's sentiment analysis
try:
    # Explicitly check for the resource file
    nltk.data.find('sentiment/vader_lexicon')
    print("NLTK 'vader_lexicon' found.")
# Catch LookupError if the resource is not found
except LookupError:
    print("NLTK 'vader_lexicon' not found. Downloading...")
    nltk.download('vader_lexicon')
    print("NLTK 'vader_lexicon' downloaded.")

print("NLTK resource check complete for testing script.")

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


# --- Load the Tokenizer configuration ---
tokenizer_config_path = 'tokenizer_config.json'
if not os.path.exists(tokenizer_config_path):
    print(f"Error: Tokenizer configuration file not found at {tokenizer_config_path}")
    print("Please run train_model.py first to generate it.")
    exit()

try:
    # Read the JSON file content as a string
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        tokenizer_json_string = f.read()

    # Pass the JSON string to tokenizer_from_json
    tokenizer = tokenizer_from_json(tokenizer_json_string)
    print(f"Tokenizer loaded successfully with vocabulary size: {len(tokenizer.word_index)}")

except Exception as e:
    print(f"Error loading tokenizer configuration: {e}")
    exit()


# --- Load the TFLite model ---
# Use the path to your LATEST saved TFLite model (optimized recommended)
# Make sure this path matches the output filename from your last train_model.py run
model_path = 'best_cnn_lstm_model_optimized_deeper.tflite' # <--- !!! UPDATED PATH !!!

if not os.path.exists(model_path):
    print(f"Error: TFLite model file not found at {model_path}")
    print(f"Please ensure '{model_path}' is in the same directory and generated by train_model.py.")
    exit()

try:
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print(f"TFLite model loaded successfully from {model_path}")

except Exception as e:
    print(f"Error loading or allocating tensors for TFLite model: {e}")
    print("Possible reasons:")
    print("- Model file is corrupted or incorrect path.")
    print("- TFLite runtime is not correctly installed.")
    "- Model uses operations (like Flex ops) not supported by the standard interpreter without delegates."
    "  If you saw Flex ops warnings during training, you might need to build/use a TFLite interpreter with Flex delegate support."
    exit()


# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get max_len and input dtype from the TFLite model's input details
max_len = input_details[0]['shape'][1]
input_dtype = input_details[0]['dtype']
print(f"Detected max_len from TFLite model: {max_len}")
print(f"Detected input dtype from TFLite model: {input_dtype}")

# --- Helper: Sentiment analysis (using TextBlob) ---
def analyze_sentiment(text):
    """Analyzes sentiment using TextBlob."""
    # TextBlob returns a Sentiment object with polarity and subjectivity
    return TextBlob(text).sentiment.polarity

# --- Define keywords associated with unhealthy items/behaviors ---
# This list helps the post-processing logic identify when sentiment is related to something unhealthy
unhealthy_keywords = [
    "pizza", "burger", "fries", "soda", "sugar", "candy", "chocolate",
    "smoking", "alcohol", "junk food", "fast food", "sedentary", "couch potato",
    "stay up late", "all night", "excessive screen time", "unhealthy food", "cigarettes",
    "binge eating", "crash diet", "skipping meals", "too much screen time", "lack of sleep"
    # Add more as needed - MUST be consistent with keywords considered in training data/logic
]

# --- Define highly indicative unhealthy keywords ---
# These keywords are strong indicators of an unhealthy habit
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
        "Content-Type": "application/json"
    }

    # Craft a prompt that encourages a helpful and persuasive response
    # Updated prompt and system message for tweet-like response
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
        print("🗣️ Persuasive Message:", message)
    except requests.exceptions.RequestException as e:
        print(f"❌ Error generating persuasive message: {e}")
        print("Please check your OpenRouter API key, network connection, and the API endpoint.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during message generation: {e}")
    print("-------------------------------------")


# --- Classifier Logic for TFLite Model with Post-processing (Warning Statement Handling) ---
def classify_text_tflite(text):
    """
    Classifies text using the loaded TFLite model and applies post-processing
    to handle warning/awareness statements, strong unhealthy keywords, and sentiment.
    If classified as unhealthy, generates a persuasive message.
    """
    print("\n--- Inside classify_text_tflite function ---") # <-- Verification Print
    original_text = text
    print(f"Original Text: '{original_text}'")

    # --- Run Model Inference ---
    # Use the SAME preprocessing function as training
    processed_text = preprocess_text(original_text)
    print(f"Processed Text for Model: '{processed_text}'")

    # Tokenize and pad the processed text
    sequences = tokenizer.texts_to_sequences([processed_text])
    # Ensure dtype matches the model's input dtype
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post', dtype=input_dtype.__name__)
    input_data = np.array(padded_sequences, dtype=input_dtype)

    # Ensure input shape matches the model's expected shape
    expected_input_shape = input_details[0]['shape']
    if len(input_data.shape) == 1:
        input_data = np.expand_dims(input_data, axis=0)

    if list(input_data.shape) != list(expected_input_shape):
        print(f"Error: Input data shape {input_data.shape} does not match expected model input shape {expected_input_shape}.")
        return None

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Extract probability for the unhealthy class
    if output_data.shape == (1, 1):
        probability_unhealthy = output_data[0][0]
    else:
        print(f"Warning: Unexpected model output shape: {output_data.shape}")
        # Attempt to handle potential different output shapes if they are probabilities
        if output_details[0]['shape'][-1] == 1:
            probability_unhealthy = output_data.flatten()[0]
            print(f"Attempted to flatten output and got probability: {probability_unhealthy:.4f}")
        else:
            print("Could not extract single probability from output.")
            return None


    probability_healthy = 1 - probability_unhealthy

    print(f"\n--- Model Prediction (Raw) ---")
    print(f"🔹 Probability (Unhealthy): {probability_unhealthy:.4f}")
    print(f"🔹 Probability (Healthy): {probability_healthy:.4f}")
    print("------------------------------")

    # --- Post-processing with Sentiment and Keywords ---
    # Use the original text for sentiment analysis as TextBlob works better on raw text
    sentiment_polarity = analyze_sentiment(original_text)
    contains_any_unhealthy_kwd = contains_keyword(original_text, unhealthy_keywords)
    contains_strong_unhealthy_kwd = contains_keyword(original_text, strong_unhealthy_keywords)
    contains_any_healthy_kwd = contains_keyword(original_text, known_healthy_habits)
    contains_avoidance_kwd = contains_keyword(original_text, avoidance_keywords) # Check for avoidance keywords
    contains_warning_kwd = contains_keyword(original_text, warning_keywords) # Check for warning keywords


    print(f"--- Post-processing Info ---")
    print(f"🔹 Sentiment Polarity: {sentiment_polarity:.4f}")
    print(f"🔹 Contains Any Unhealthy Keyword: {contains_any_unhealthy_kwd}")
    print(f"🔹 Contains Strong Unhealthy Keyword: {contains_strong_unhealthy_kwd}")
    print(f"🔹 Contains Any Healthy Keyword: {contains_any_healthy_kwd}")
    print(f"🔹 Contains Avoidance Keyword: {contains_avoidance_kwd}")
    print(f"🔹 Contains Warning Keyword: {contains_warning_kwd}") # Debug print
    print("--------------------------")

    # Define thresholds for post-processing
    # These thresholds can be tuned based on desired behavior
    model_high_confidence_threshold = 0.8 # Higher threshold for strong model conviction
    negative_sentiment_threshold = -0.1
    positive_sentiment_threshold = 0.1
    model_default_threshold = 0.5 # Standard threshold for model's raw prediction
    strong_override_sentiment_threshold = 0.5 # Higher sentiment threshold for overriding strong keywords
    # Threshold for negative sentiment override when unhealthy keyword and avoidance keyword are present
    negative_sentiment_unhealthy_avoidance_override_threshold = -0.3 # Tune this value


    print(f"--- Debugging Post-processing Conditions ---")
    print(f"Thresholds: High Conf={model_high_confidence_threshold}, Neg Sent={negative_sentiment_threshold}, Pos Sent={positive_sentiment_threshold}, Default Model={model_default_threshold}, Strong Override Sent={strong_override_sentiment_threshold}, Neg Sent Unhealthy Avoidance Override={negative_sentiment_unhealthy_avoidance_override_threshold}")
    print(f"Model Prob Unhealthy: {probability_unhealthy:.4f}, Prob Healthy: {probability_healthy:.4f}")
    print(f"Sentiment Polarity: {sentiment_polarity:.4f}")
    print(f"Contains Any Unhealthy Kwd: {contains_any_unhealthy_kwd}, Strong Unhealthy Kwd: {contains_strong_unhealthy_kwd}, Any Healthy Kwd: {contains_any_healthy_kwd}, Avoidance Kwd: {contains_avoidance_kwd}, Warning Kwd: {contains_warning_kwd}")


    final_classification = "healthy" # Default initial classification

    # --- Further Revised Post-processing Logic ---
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
        # (e.g., expressing strong relief/joy about quitting a strong unhealthy habit)
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
        # Sub-Rule 2a: If sentiment is significantly negative AND contains avoidance keywords, classify as HEALTHY (talking negatively about unhealthy in a healthy way)
        if (sentiment_polarity < negative_sentiment_unhealthy_avoidance_override_threshold and
                contains_avoidance_kwd):
            final_classification = "healthy"
            print("✅ Post-processing override (Rule 2a - Negative Sentiment + Avoidance on Unhealthy): Any unhealthy keyword + Significantly negative sentiment + Avoidance keyword -> Classified as **HEALTHY**.")
        # If model is highly confident it's unhealthy, confirm UNHEALTHY (unless overridden by 2a)
        elif probability_unhealthy > model_high_confidence_threshold:
            final_classification = "unhealthy"
            print("🚨 Post-processing Rule 2b: Any unhealthy keyword + High unhealthy probability -> Classified as **UNHEALTHY**.")
        # If model is highly confident it's healthy AND sentiment is negative, override to UNHEALTHY (less likely with 2a)
        elif probability_healthy > model_high_confidence_threshold and sentiment_polarity < negative_sentiment_threshold:
            final_classification = "unhealthy"
            print("🚨 Post-processing override (Rule 2c - Override Any Unhealthy): Any unhealthy keyword + High healthy prob + Negative sentiment -> Classified as **UNHEALTHY**.")
        # Otherwise, rely on the model's default prediction for these cases
        else:
            if probability_unhealthy > model_default_threshold:
                final_classification = "unhealthy"
                print("🚨 Model Classified (Rule 2d Default): Any unhealthy keyword present, Model Classified **UNHEALTHY**.")
            else:
                final_classification = "healthy"
                print("✅ Model Classified (Rule 2d Default): Any unhealthy keyword present, Model Classified **HEALTHY**.")


    # Rule 3: If ANY healthy keyword is present (and no unhealthy keywords or warnings, handled by Rule 0/1/2)
    elif contains_any_healthy_kwd:
        # If model is highly confident it's healthy, confirm HEALTHY
        if probability_healthy > model_high_confidence_threshold:
            final_classification = "healthy"
            print("✅ Post-processing Rule 3a: Any healthy keyword + High healthy probability -> Classified as **HEALTHY**.")
        # If model is highly confident it's unhealthy AND sentiment is positive, override to HEALTHY
        elif probability_unhealthy > model_high_confidence_threshold and sentiment_polarity > positive_sentiment_threshold:
            final_classification = "healthy"
            print("✅ Post-processing override (Rule 3b - Override Any Healthy): Any healthy keyword + High unhealthy prob + Positive sentiment -> Classified as **HEALTHY**.")
        # Otherwise, rely on the model's default prediction for these cases
        else:
            if probability_unhealthy > model_default_threshold:
                final_classification = "unhealthy"
                print("🚨 Model Classified (Rule 3c Default): Any healthy keyword present, Model Classified **UNHEALTHY**.")
            else:
                final_classification = "healthy"
                print("✅ Model Classified (Rule 3c Default): Any healthy keyword present, Model Classified **HEALTHY**.")

    # Rule 4: If NO relevant keyword is found (and not a warning, handled by Rule 0)
    else:
        # Rely solely on the model's default prediction
        if probability_unhealthy > model_default_threshold:
            final_classification = "unhealthy"
            print("🚨 Model Classified (Rule 4 - Default): No relevant keyword, Model Classified **UNHEALTHY**.")
        else:
            final_classification = "healthy"
            print("✅ Model Classified (Rule 4 - Default): No relevant keyword, Model Classified **HEALTHY**.")


    print("--------------------------------------------")

    # --- Generate persuasive message if classified as UNHEALTHY ---
    if final_classification == "unhealthy":
        generate_persuasive_message(original_text)

    return final_classification

# --- Interactive loop ---
print("\nHealth Advisor Bot - Local Model Test with Further Revised Post-processing and Persuasive Messages")
print("Type 'exit' to quit.")
while True:
    habit = input("\nEnter a habit: ").strip()
    if habit.lower() == "exit":
        print("Goodbye!")
        break
    if not habit:
        continue

    classification = classify_text_tflite(habit)
    # The classification and reasoning are already printed inside the function
