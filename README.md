# Memotion Sentiment Analysis

This project implements a **Multi-Modal Deep Learning Architecture** to classify meme sentiments. The model processes three distinct data inputs—**Images, Text, and Tabular Metadata**—simultaneously to predict the 'Overall Sentiment' of memes.

## 📋 Project Overview
The pipeline covers the entire lifecycle, from data preprocessing to building a sophisticated hybrid neural network:

1. **Data Cleaning:** Handling nulls, filtering to 1,500 samples, and preprocessing text.
2. **Feature Engineering:** Encoding tabular features via `OneHotEncoder` and cleaning meme text for sequence modeling.
3. **Hybrid Modeling:** Constructing a multi-input neural network.

## 🏗 Model Architecture
The model uses a fusion-based design where three parallel branches extract features before concatenating them for final classification:

*   **Image Branch (CNN):** Uses `Conv2D` layers with `BatchNormalization` and `Dropout` to extract spatial features, reduced by `GlobalAveragePooling2D`.
*   **Text Branch (NLP):** Processes text sequences through an `Embedding` layer, followed by a `Bidirectional LSTM` and a `GRU` layer to capture temporal dependencies.
*   **Tabular Branch:** A simple deep dense network processing numerical/categorical metadata.
*   **Fusion Layer:** Combines the outputs of the image, text, and tabular branches using `Concatenate`, followed by `Dense` layers with `PReLU` activation.
*   **Output:** A 5-class `softmax` layer for sentiment prediction.

**Technical Configuration:**
- **Optimizer:** Adam (Learning Rate: 0.002)
- **Loss:** Sparse Categorical Crossentropy

## 🛠 Technologies & Libraries
- **Deep Learning:** TensorFlow/Keras
- **Data Manipulation:** Pandas, NumPy, Scikit-learn
- **NLP:** Keras Tokenizer, Pad Sequences, RegEx
- **Visualization:** Matplotlib, Seaborn
