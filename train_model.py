import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
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
except LookupError:
    print("NLTK 'omw-1.4' not found. Downloading...")
    nltk.download('omw-1.4')
    print("NLTK 'omw-1.4' downloaded.")
print("NLTK resource check complete.")

# --- Custom Stopword Set ---
stop_words = set(stopwords.words('english'))
sentiment_words_to_keep = {'love', 'hate', 'like', 'dislike', 'good', 'bad', 'not', 'very', 'too', 'less', 'more', 'no',
                           'healthy', 'unhealthy', 'sick', 'well', 'pain', 'food', 'eat', 'drink', 'exercise', 'stress',
                           'never', 'none', 'nothing', 'nowhere', 'hardly', 'scarcely', 'barely', 'aint', 'isnt', 'arent',
                           'wasnt', 'werent', 'hasnt', 'havent', 'hadnt', 'wont', 'wouldnt', 'dont', 'doesnt', 'didnt',
                           'cant', 'couldnt', 'shouldnt', 'mightnt', 'mustnt'}
stop_words = stop_words - sentiment_words_to_keep

# --- Lemmatizer Initialization ---
lemmatizer = WordNetLemmatizer()

# --- Enhanced Preprocessing function ---
def preprocess_text_enhanced(text):
    """
    Cleans and preprocesses text: lowercases, removes URLs, mentions, hashtags, punctuation,
    tokenizes, handles simple negation, removes custom stopwords, and lemmatizes.
    """
    if pd.isna(text):
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
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    print("Embedding matrix created.")
    return embedding_matrix

# --- Configuration ---
dataset_path = 'final_dataset.csv'
glove_file = 'path/to/your/glove.6B.100d.txt' # <--- !!! CHANGE THIS PATH to your GloVe file !!!
embedding_dim = 100
max_words = 10000
epochs = 50
batch_size = 64
validation_split_ratio = 0.15
test_size_ratio = 0.2

# --- 1. Load the dataset ---
if not os.path.exists(dataset_path):
    print(f"Error: Dataset file not found at {dataset_path}")
    print("Please ensure 'final_dataset.csv' is in the same directory.")
    exit()

df = pd.read_csv(dataset_path)
df['text'] = df['text'].astype(str).fillna("")

# --- 2. Separate text and labels ---
texts = df['text']
labels = df['label']

# --- 3. Preprocess the text data ---
print("Preprocessing text data with enhancements...")
processed_texts = texts.apply(preprocess_text_enhanced)
print("Preprocessing complete.")

# --- 4. Encode the labels (healthy -> 0, unhealthy -> 1) ---
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"Original labels: {label_encoder.classes_}")
print(f"Encoded labels mapping: {list(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# --- 5. Split the data into training and testing sets ---
train_texts, test_texts, train_labels, test_labels = train_test_split(
    processed_texts, encoded_labels, test_size=test_size_ratio, random_state=42, stratify=encoded_labels
)
print(f"Training samples: {len(train_texts)}")
print(f"Testing samples: {len(test_texts)}")

# --- 6. Analyze tweet lengths and set max_len ---
train_text_lengths = [len(text.split()) for text in train_texts]
max_len = int(np.percentile(train_text_lengths, 95))
print(f"Setting max_len to: {max_len}") # Make a note of this value for the API script!

# --- 7. Tokenize the text data ---
tokenizer = Tokenizer(num_words=max_words, oov_token="<unk>")
tokenizer.fit_on_texts(train_texts)

# Save the tokenizer configuration
tokenizer_config_path = 'tokenizer_config.json'
tokenizer_config = tokenizer.to_json()
with open(tokenizer_config_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(json.loads(tokenizer_config), indent=4))
print(f"Tokenizer configuration saved to {tokenizer_config_path}")

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# --- 8. Pad the sequences to have the same length ---
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

# Embedding Layer
if use_pretrained_embeddings:
    model.add(Embedding(
        max_words,
        embedding_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=True
    ))
else:
    model.add(Embedding(
        max_words,
        embedding_dim,
        input_length=max_len,
        trainable=True
    ))

model.add(Dropout(0.4))

# CNN Layer
model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
model.add(Dropout(0.4))
model.add(MaxPooling1D(pool_size=4))

# First Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(64, return_sequences=True, recurrent_dropout=0.3)))
model.add(Dropout(0.4))

# Second Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(64, recurrent_dropout=0.3)))

model.add(Dropout(0.4))

# Dense output layer
model.add(Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.001)))

# --- 10. Compile the model ---
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# --- 11. Train the model with Advanced Callbacks ---
print("Training model with Early Stopping, Model Checkpointing, and ReduceLROnPlateau...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    'best_cnn_lstm_model_deeper.keras', # Saves the model in .keras format
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=0.0001,
    verbose=1
)

history = model.fit(
    padded_train_sequences,
    train_labels,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split_ratio,
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)
print("Training complete.")

# Load the best model saved by ModelCheckpoint for evaluation
try:
    # Ensure this path matches the filename in ModelCheckpoint
    best_model = tf.keras.models.load_model('best_cnn_lstm_model_deeper.keras')
    print("Loaded best deeper regularized model for evaluation.")
    model_to_evaluate = best_model
except Exception as e:
    print(f"Could not load best deeper regularized model: {e}. Evaluating the last trained model.")
    model_to_evaluate = model

# --- 12. Evaluate the model ---
print("\nEvaluating model on test set...")
loss, accuracy = model_to_evaluate.evaluate(padded_test_sequences, test_labels, verbose=0)
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

# The TFLite conversion section is removed.
# The model is saved as 'best_cnn_lstm_model_deeper.keras' by the ModelCheckpoint.
print("\nTraining script finished. The best model is saved as 'best_cnn_lstm_model_deeper.keras'.")
