import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Maximum number of words to consider in the vocabulary
max_features = 50000
# Maximum length of input sequences
maxlen = 100

# Load the IMDb dataset with the specified vocabulary size
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad sequences to ensure all inputs have the same length
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# Get the word-to-index mapping
word_index = imdb.get_word_index()

# Create a reverse mapping (index-to-word) to decode encoded reviews
reverse_word_index = {index + 3: word for word, index in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"
reverse_word_index[3] = "<UNUSED>"

# Function to decode an encoded review back to text
def decode_review(encoded_review):
    return " ".join([reverse_word_index.get(i, "?") for i in encoded_review])

# Randomly select 3 reviews from the training dataset to display
random_indices = np.random.choice(len(x_train), size=3, replace=False)
for i in random_indices:
    print(f"Comment: {decode_review(x_train[i])}")  # Decode and print the review
    print(f"Label: {y_train[i]}\n")  # Print the sentiment label (0 = negative, 1 = positive)

# Transformer block definition
class TransformerBlock(layers.Layer):
    def __init__(self, embed_size, heads, dropout_rate=0.3):
        super(TransformerBlock, self).__init__()
        
        # Multi-head self-attention layer
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=embed_size)
        # Layer normalization for attention output
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        # Layer normalization for feed-forward output
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network
        self.feed_forward = models.Sequential([
            layers.Dense(embed_size * 4, activation="relu"),  # Hidden layer
            layers.Dense(embed_size, activation="relu"),      # Output layer
        ])
        
        # Dropout layers to prevent overfitting
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
    
    # Forward pass
    def call(self, x, training):
        attention = self.attention(x, x)  # Self-attention
        # Residual connection and normalization
        x = self.norm1(x + self.dropout1(attention, training=training))
        # Feed-forward network
        feed_forward = self.feed_forward(x)
        # Residual connection and normalization
        return self.norm2(x + self.dropout2(feed_forward, training=training))

# Transformer-based sentiment analysis model definition
class TransformerModel(models.Model):
    def __init__(self, num_layers, embed_size, heads, input_dim, output_dim, dropout_rate=0.1):
        super(TransformerModel, self).__init__()

        # Embedding layer for input sequences
        self.embedding = layers.Embedding(input_dim=input_dim, output_dim=embed_size)
        # Stack of transformer blocks
        self.transformer_blocks = [
            TransformerBlock(embed_size, heads, dropout_rate) for _ in range(num_layers)
        ]
        # Global average pooling for reducing sequence to a fixed size
        self.global_avg_pooling = layers.GlobalAveragePooling1D()
        # Dropout layer for regularization
        self.dropout = layers.Dropout(dropout_rate)
        # Fully connected layer for classification
        self.fc = layers.Dense(output_dim, activation="sigmoid")

    # Forward pass
    def call(self, x, training=False):
        x = self.embedding(x)  # Embedding the input
        for transformer in self.transformer_blocks:
            x = transformer(x, training=training)  # Pass through transformer blocks
        x = self.global_avg_pooling(x)  # Pool the sequence
        x = self.dropout(x, training=training)  # Apply dropout
        return self.fc(x)  # Final classification layer

# Model hyperparameters
num_layers = 4
embed_size = 64
num_heads = 4
input_dim = max_features
output_dim = 1
dropout_rate = 0.1

# Initialize the model
model = TransformerModel(num_layers, embed_size, num_heads, input_dim, output_dim, dropout_rate)

# Build the model by specifying input shape
model.build(input_shape=(None, maxlen))

# Compile the model with optimizer, loss, and metrics
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Print the model architecture
model.summary()

# Train the model with training data and validate with test data
history = model.fit(x_train, y_train, epochs=5, batch_size=256, validation_data=(x_test, y_test))

# Plot training and validation loss
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()

# Function to predict sentiment of a user-provided review
def predict_sentiment(model, text, word_index, maxlen):
    # Encode the review text into numerical indices
    encoded_text = [word_index.get(word, 0) for word in text.lower().split()]
    # Pad the encoded sequence to match input length
    padded_text = pad_sequences([encoded_text], maxlen=maxlen)
    # Predict sentiment using the trained model
    prediction = model.predict(padded_text)
    return prediction[0][0]

# Get word-to-index mapping again
word_index = imdb.get_word_index()
# Take user input for a review
user_input = input("Enter a film review: ")
# Predict the sentiment of the review
sentiment_score = predict_sentiment(model, user_input, word_index, maxlen)

# Print the sentiment result based on the prediction score
if sentiment_score > 0.5:
    print(f"Prediction result is positive -> score: {sentiment_score}")
else:
    print(f"Prediction result is negative -> score: {sentiment_score}")
