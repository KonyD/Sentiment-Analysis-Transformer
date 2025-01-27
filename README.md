# Sentiment Analysis with Transformer

This project implements a sentiment analysis model using a custom Transformer architecture. The model predicts whether a movie review is **positive** or **negative** based on the IMDB dataset.

## 🚀 Features

- **Custom Transformer Model**: Built from scratch using TensorFlow and Keras.
- **IMDB Dataset**: Preprocessed and padded for consistency.
- **Visualization**: Includes loss and accuracy plots for training and validation.

## 📂 Project Structure

```plaintext
.
├── main.py            # Main script with the Transformer model
├── requirements.txt   # Dependencies for the project
├── .gitignore         # Ignored files for Git
└── README.md          # Project documentation
```

## 🛠️ Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

## 🧪 Usage

1. Run the main script:

   ```bash
   python main.py
   ```

2. Enter a movie review when prompted, and the model will predict its sentiment.

## 📊 Visualization

Training and validation metrics are plotted after training:

- **Loss vs Epochs**
![Training Loss and Validation Loss](./Figure%202025-01-27%20165720.png)
- **Accuracy vs Epochs**
![Training Loss and Validation Loss](./Figure%202025-01-27%20165742.png)

## ✍️ Example Output

```
Enter a film review: This movie was amazing! I loved it.
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 63ms/step
Prediction result is positive -> score: 0.9998332262039185
```

## 📈 Model Summary

The model uses a 4-layer Transformer with multi-head attention and feedforward layers, optimized for binary classification.

## 📋 Dataset

The project uses the IMDB dataset, which is loaded via TensorFlow's `keras.datasets.imdb`.

## 🤝 Contributing

Feel free to fork the repository, submit pull requests, or open issues to contribute to this project.

---
