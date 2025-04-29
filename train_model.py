import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional # Added Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau # Added ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # Added WordNetLemmatizer
import json
import os

# --- NLTK Downloads ---
import nltk
print("Checking for NLTK resources...")
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
print("NLTK resource check complete.")


# --- Custom Stopword Set ---
# Start with the standard English stopwords
stop_words = set(stopwords.words('english'))

# Define words that should NOT be removed, even if they are in the standard stopword list
# These are words that might be important for sentiment or context in your health domain
# Added more potential sentiment/context words relevant to health/unhealthy
sentiment_words_to_keep = {'love', 'hate', 'like', 'dislike', 'good', 'bad', 'not', 'very', 'too', 'less', 'more', 'no',
                           'healthy', 'unhealthy', 'sick', 'well', 'pain', 'food', 'eat', 'drink', 'exercise', 'stress',
                           'never', 'none', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'aint', 'isnt', 'arent',
                           'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wont', 'wouldnt', 'dont', 'doesnt', 'didnt',
                           'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'} # Added more negation-related words

# Remove these sentiment words from the stopword set
stop_words = stop_words - sentiment_words_to_keep

# --- Lemmatizer Initialization ---
lemmatizer = WordNetLemmatizer()

# --- Enhanced Preprocessing function ---
def preprocess_text_enhanced(text):
    """
    Cleans and preprocesses text: lowercases, removes URLs, mentions, hashtags, punctuation,
    tokenizes, handles simple negation, removes custom stopwords, and lemmatizes.
    """
    if pd.isna(text): # Handle potential NaN values in text column
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs more robustly
    text = re.sub(r'@\w+', '', text) # Remove mentions (@user)
    text = re.sub(r'#', '', text) # Remove hashtags (#) - keeping the word, removing the symbol
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation

    word_tokens = word_tokenize(text)

    # Simple Negation Handling: Join negation words with the next word
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

# --- Function to load pre-trained embeddings ---
def load_glove_embeddings(glove_file_path, embedding_dim):
    """Loads GloVe embeddings from a file."""
    embeddings_index = {}
    print(f"Loading GloVe embeddings from {glove_file_path}...")
    try:
        with open(glove_file_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print(f"Successfully loaded {len(embeddings_index)} word vectors.")
        return embeddings_index
    except FileNotFoundError:
        print(f"Error: GloVe embedding file not found at {glove_file_path}")
        print("Please download GloVe embeddings (e.g., from https://nlp.stanford.edu/projects/glove/) and update the path.")
        return None
    except Exception as e:
        print(f"Error loading GloVe embeddings: {e}")
        return None

# --- Function to create embedding matrix ---
def create_embedding_matrix(embeddings_index, word_index, embedding_dim, max_words):
    """Creates an embedding matrix from loaded embeddings and tokenizer's word index."""
    # +1 because word_index is 1-based indexing in Keras Tokenizer
    embedding_matrix = np.zeros((max_words, embedding_dim))
    # The 0th index is reserved for padding in Keras Embedding layer, so we start from 1
    for word, i in word_index.items():
        if i < max_words: # Ensure index is within our max_words vocabulary size
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all zeros (handled by np.zeros initialization).
                embedding_matrix[i] = embedding_vector
    print("Embedding matrix created.")
    return embedding_matrix

# --- Configuration ---
dataset_path = 'final_dataset.csv'
glove_file = 'path/to/your/glove.6B.100d.txt' # <--- !!! CHANGE THIS PATH to your GloVe file !!!
embedding_dim = 100 # Make sure this matches the dimension of your GloVe file (e.g., 100d, 200d, 300d)
max_words = 10000 # Increased vocabulary size slightly
epochs = 50 # Increased epochs as regularization might slow down training convergence
batch_size = 64 # Increased batch size slightly
validation_split_ratio = 0.15 # Use 15% of training data for validation
test_size_ratio = 0.2 # 20% for testing

# --- 1. Load the dataset ---
if not os.path.exists(dataset_path):
    print(f"Error: Dataset file not found at {dataset_path}")
    print("Please ensure 'final_dataset.csv' is in the same directory.")
    exit()

df = pd.read_csv(dataset_path)

# Ensure text column is string type and handle potential NaNs
df['text'] = df['text'].astype(str).fillna("")

# --- 2. Separate text and labels ---
texts = df['text']
labels = df['label']

# --- 3. Preprocess the text data ---
print("Preprocessing text data with enhancements...")
# Apply the enhanced preprocessing function
processed_texts = texts.apply(preprocess_text_enhanced)
print("Preprocessing complete.")

# --- 4. Encode the labels (healthy -> 0, unhealthy -> 1) ---
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_) # Should be 2 for binary

print(f"Original labels: {label_encoder.classes_}")
print(f"Encoded labels mapping: {list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")


# --- 5. Split the data into training and testing sets ---
# Use stratify to maintain label distribution
train_texts, test_texts, train_labels, test_labels = train_test_split(
    processed_texts, encoded_labels, test_size=test_size_ratio, random_state=42, stratify=encoded_labels
)
print(f"Training samples: {len(train_texts)}")
print(f"Testing samples: {len(test_texts)}")

# --- 6. Analyze tweet lengths and set max_len ---
train_text_lengths = [len(text.split()) for text in train_texts]
# Use a percentile to determine max_len, ensuring most texts are covered
max_len = int(np.percentile(train_text_lengths, 95)) # Increased percentile slightly for potentially longer texts
print(f"Setting max_len to: {max_len}")

# --- 7. Tokenize the text data ---
tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
tokenizer.fit_on_texts(train_texts) # Fit on processed training texts

# Save the tokenizer configuration
tokenizer_config_path = 'tokenizer_config.json'
tokenizer_config = tokenizer.to_json()
with open(tokenizer_config_path, 'w', encoding='utf-8') as f: # Specify encoding
    f.write(json.dumps(json.loads(tokenizer_config), indent=4)) # Prettier formatting
print(f"Tokenizer configuration saved to {tokenizer_config_path}")

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# --- 8. Pad the sequences to have the same length ---
# Ensure dtype is float32 for TFLite compatibility later
padded_train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post', truncating='post', dtype='float32')
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post', truncating='post', dtype='float32')

# --- Load Pre-trained Embeddings and Create Embedding Matrix ---
embeddings_index = load_glove_embeddings(glove_file, embedding_dim)

use_pretrained_embeddings = False
embedding_matrix = None

if embeddings_index is not None:
    embedding_matrix = create_embedding_matrix(embeddings_index, tokenizer.word_index, embedding_dim, max_words)
    use_pretrained_embeddings = True
    print("Using pre-trained GloVe embeddings.")
else:
    print("GloVe embeddings not found or failed to load. Training embeddings from scratch.")


# --- 9. Build the CNN-LSTM model (Deeper Bi-LSTM) ---
print("Building CNN-LSTM model (Deeper Bi-LSTM)...")
model = Sequential()

# Embedding Layer (Use pre-trained if available, and make trainable)
if use_pretrained_embeddings:
    model.add(Embedding(
        max_words, # Vocabulary size
        embedding_dim,
        weights=[embedding_matrix], # Use pre-trained weights
        input_length=max_len,
        trainable=True # Set to True to fine-tune embeddings on your data
    ))
else:
    model.add(Embedding(
        max_words, # Vocabulary size
        embedding_dim,
        input_length=max_len,
        trainable=True # Train embeddings from scratch
    ))

model.add(Dropout(0.4)) # Dropout after embedding

# CNN Layer to extract local features
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(Dropout(0.4)) # Added Dropout after Conv1D
model.add(MaxPooling1D(pool_size=4)) # Pool the features


# First Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.3))) # Added return_sequences=True
model.add(Dropout(0.4)) # Dropout after the first Bi-LSTM

# Second Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(64, recurrent_dropout=0.3))) # Second Bi-LSTM

model.add(Dropout(0.4)) # Dropout after the second Bi-LSTM

# Dense output layer for binary classification with L2 regularization
model.add(Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001))) # Added L2 regularization

# --- 10. Compile the model ---
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary() # Print model summary

# --- 11. Train the model with Advanced Callbacks ---
print("Training model with Early Stopping, Model Checkpointing, and ReduceLROnPlateau...")

# Define Advanced Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=7,         # Increased patience slightly
    restore_best_weights=True # Restore model weights from the epoch with best val_loss
)

model_checkpoint = ModelCheckpoint(
    'best_cnn_lstm_model_deeper.keras', # Path to save the best model (TensorFlow SavedModel format)
    monitor='val_loss',          # Monitor validation loss
    save_best_only=True,         # Save only the best model
    mode='min',                  # We want to minimize the validation loss
    verbose=1                    # Print messages when saving
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', # Monitor validation loss
    factor=0.5,         # Reduce learning rate by half
    patience=4,         # If val_loss doesn't improve for 4 epochs (adjusted with increased ES patience)
    min_lr=0.0001,      # Minimum learning rate
    verbose=1
)

# Fit the model
history = model.fit(
    padded_train_sequences,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split_ratio, # Use a portion of training data for validation
    callbacks=[early_stopping, model_checkpoint, reduce_lr] # Add the advanced callbacks
)
print("Training complete.")

# Load the best model saved by ModelCheckpoint before evaluation
try:
    best_model = tf.keras.models.load_model('best_cnn_lstm_model_deeper.keras')
    print("Loaded best deeper regularized model for evaluation.")
    model_to_evaluate = best_model
except Exception as e:
    print(f"Could not load best deeper regularized model: {e}. Evaluating the last trained model.")
    model_to_evaluate = model


# --- 12. Evaluate the model ---
print("\nEvaluating model on test set...")
loss, accuracy = model_to_evaluate.evaluate(padded_test_sequences, test_labels, verbose=0) # Set verbose to 0 to suppress progress bar
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- 13. Make predictions on the test set ---
print("Generating predictions...")
predictions = model_to_evaluate.predict(padded_test_sequences)
binary_predictions = (predictions > 0.5).astype(int)

# --- 14. Generate classification report ---
print("\nClassification Report:")
print(classification_report(test_labels, binary_predictions, target_names=label_encoder.classes_))

# --- 15. Generate confusion matrix ---
cm = confusion_matrix(test_labels, binary_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()

# --- 16. Plot training history (accuracy and loss) ---
print("Plotting training history...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# --- 17. Convert the best model to TensorFlow Lite with Optimization (Quantization) ---
print("\nConverting the best deeper regularized model to TensorFlow Lite...")

# Use the best model loaded by ModelCheckpoint for conversion
converter = tf.lite.TFLiteConverter.from_keras_model(model_to_evaluate)

# Enable optimization (default quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Specify the supported operations (TFLite Builtins and Select TF Ops)
# SELECT_TF_OPS is needed if your model uses any TensorFlow operations not natively supported by TFLite
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# This flag might be needed depending on your TensorFlow version and layers used
# converter._experimental_lower_tensor_list_ops = False # Uncomment if you face conversion issues

try:
    tflite_model = converter.convert()
    model_tflite_path = 'best_cnn_lstm_model_optimized_deeper.tflite' # Changed filename
    # Save the TFLite model
    with open(model_tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Optimized TensorFlow Lite model saved as {model_tflite_path}")

except Exception as e:
    print(f"Error during TFLite conversion with optimization: {e}")
    print("Attempting conversion without optimization as a fallback.")
    # Attempt conversion without optimization if the first one fails
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model_to_evaluate)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        # converter._experimental_lower_tensor_list_ops = False # Uncomment if needed
        tflite_model_no_opt = converter.convert()
        model_tflite_path_no_opt = 'best_cnn_lstm_model_no_optimization_deeper.tflite' # Changed filename
        with open(model_tflite_path_no_opt, 'wb') as f:
            f.write(tflite_model_no_opt)
        print(f"TensorFlow Lite model (without optimization) saved as {model_tflite_path_no_opt}")
        print("Note: The unoptimized model will be larger and potentially slower.")
    except Exception as e_no_opt:
        print(f"Error during TFLite conversion without optimization: {e_no_opt}")
        print("TFLite conversion failed.")
