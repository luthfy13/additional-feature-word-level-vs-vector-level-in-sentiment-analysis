import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import pad_sequences
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPool1D, Input, Multiply, Add,
    Concatenate, Flatten, Activation, RepeatVector, Permute, Reshape)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from keras.regularizers import l2
from gensim.models import Word2Vec
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
from tqdm import tqdm
import pickle

# negation handling function
from NegationHandlingBaseline import negation_handling as nh_baseline, custom_split

# FIXED: Using PRDECT-ID dataset only (DATASET_ID = 0)
DATASET_ID = 0
IS_PARTITIONED_DATA = 1
MAX_SEQUENCE_LENGTH = 185
BATCH_SIZE = 32  # Good balance for this dataset size
EPOCHS = 20  # Increased for more training time
EMBEDDING_DIMENSIONS = 200  # Good for longer sequences
NEGATION_EMBEDDING_DIMENSIONS = 16  # Increased for better negation representation
LSTM_UNITS = 128  # Adequate for this dataset
DROPOUT_RATE = 0.5  # Increased to prevent overfitting
INITIAL_LEARNING_RATE = 3e-4  # Slightly lower for more stable training
FINAL_LEARNING_RATE = 3e-6  # 100x reduction over training
L2_REGULARIZATION = 1e-4  # Keep as is
EARLY_STOPPING_PATIENCE = 5  # More time to converge

# 0 : konkatenasi embedding
# 1 : Feature Interaction
# 2 : konkatenasi + attention mechanism
# 3 : Focused Attention Mechanism
EMBEDDING_INTEGRATION_MODE = 0

# Comparison Mode
TRAIN_BOTH = True  # Set True to compare vector-level vs word-level
USE_WORD_LEVEL = False  # Only used if TRAIN_BOTH = False

VALIDATION_SPLIT              = 0.0
REDUCE_LR_PATIENCE            = 3
REDUCE_LR_FACTOR              = 0.5
RANDOM_STATE                  = 42

# Paths
WORD2VEC_MODEL_PATH = "..\\data\\vector-models\\model.model"
MODEL_SAVE_PATH = "models/"
RESULTS_SAVE_PATH = "results/"

ZERO_VECTOR = True

# Ensure directories exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)

def train_word2vec_model(train_df, model_save_path=None):
    """
    Train a Word2Vec model from scratch using the dataset

    Args:
        train_df: pandas DataFrame with 'text' column
        model_save_path: path to save the trained model (optional)

    Returns:
        word2vec_model: trained Word2Vec model
    """
    print("Training Word2Vec model from scratch...")

    # Prepare sentences for training
    sentences = []
    for text in tqdm(train_df['text'].values, desc="Tokenizing texts"):
        # Use the same tokenization as in the rest of the pipeline
        tokens = custom_split(text.lower())
        sentences.append(tokens)

    # Train Word2Vec model
    word2vec_model = Word2Vec(
        sentences=sentences,
        vector_size=EMBEDDING_DIMENSIONS,  # Match existing EMBEDDING_DIMENSIONS
        window=8,             # Context window size
        min_count=2,          # Ignore words with frequency below this
        sg=1,                 # Use skip-gram (1) vs CBOW (0)
        workers=4,            # Number of CPU cores to use
        epochs=10,            # Number of training epochs
        seed=RANDOM_STATE     # For reproducibility
    )

    # Save the model if a path is provided
    if model_save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        word2vec_model.save(model_save_path)
        print(f"Word2Vec model saved to {model_save_path}")

    # Print vocabulary size
    print(f"Word2Vec vocabulary size: {len(word2vec_model.wv)}")

    return word2vec_model

def load_data_partitioned():
    """
    Load data from CSV files - PRDECT-ID dataset only
    """
    print(f"\nusing prdct-id dataset")
    print(f"======================\n")
    train_df = pd.read_csv('data/dataset/partitioned/train-prdct-id.csv')
    validation_df = pd.read_csv('data/dataset/partitioned/val-prdct-id.csv')
    test_df = pd.read_csv('data/dataset/partitioned/test-prdct-id.csv')

    print(f"Partitioned Data Loaded")

    print(f"Train data shape: {train_df.shape}")
    print(f"Validation data shape: {validation_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Check class distribution
    print("\nClass distribution:")
    print("Train data:")
    print(train_df['sentiment'].value_counts(normalize=True))
    print("Validation data:")
    print(validation_df['sentiment'].value_counts(normalize=True))
    print("Test data:")
    print(test_df['sentiment'].value_counts(normalize=True))

    return train_df, validation_df, test_df

def preprocess_data(dataframe, word2vec_model, with_negation=False, negation_approach=5):
    """
    Preprocess data for model training

    Args:
        dataframe: pandas DataFrame with 'text' and 'sentiment' columns
        word2vec_model: trained Word2Vec model
        with_negation: whether to use negation handling
        negation_approach: which negation approach to use (5 for FWL baseline)

    Returns:
        X: preprocessed features
        y: labels
        negation_vectors: negation vectors (if with_negation=True)
    """
    texts = dataframe['text'].values
    labels = dataframe['sentiment'].values

    word_index = {word: i+1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
    vocab_size = len(word_index) + 1  # Add 1 for OOV token

    all_sequences = []
    all_negation_vectors = []

    for text in tqdm(texts, desc="Processing texts"):
        if with_negation:
            # Get tokens and negation vector using the baseline approach (FWL = baseline 5)
            tokens, negation_vector = nh_baseline(text, negation_approach)

            # Ensure negation vector has same length as tokens
            if len(negation_vector) != len(tokens):
                # Trim or pad as needed
                negation_vector = negation_vector[:len(tokens)] if len(negation_vector) > len(tokens) else negation_vector + [0] * (len(tokens) - len(negation_vector))

            # Convert tokens to sequence of word indices
            sequence = []
            for token in tokens:
                if token in word_index:
                    sequence.append(word_index[token])
                else:
                    sequence.append(0)  # OOV token

            all_sequences.append(sequence)
            all_negation_vectors.append(negation_vector)
        else:
            # Simple tokenization if not using negation handling
            tokens = custom_split(text.lower())
            sequence = []
            for token in tokens:
                if token in word_index:
                    sequence.append(word_index[token])
                else:
                    sequence.append(0)  # OOV token

            all_sequences.append(sequence)

    # Pad sequences to MAX_SEQUENCE_LENGTH
    X = pad_sequences(all_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    y = labels

    if with_negation:
        negation_vectors = pad_sequences(all_negation_vectors, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        return X, y, negation_vectors
    else:
        return X, y

def preprocess_data_word_level(dataframe, word2vec_model, negation_approach=5):
    """
    Preprocess data with WORD-LEVEL negation augmentation (NOT_ tagging)

    Args:
        dataframe: pandas DataFrame with 'text' and 'sentiment' columns
        word2vec_model: trained Word2Vec model
        negation_approach: which negation approach to use (5 for FWL baseline)

    Returns:
        X: preprocessed features (word indices with NOT_ tokens)
        y: labels
        vocab_stats: dictionary with vocabulary statistics
    """
    texts = dataframe['text'].values
    labels = dataframe['sentiment'].values

    # Build vocabulary including NOT_ variants
    all_tagged_tokens = []
    not_word_count = 0

    for text in tqdm(texts, desc="Building vocabulary with NOT_ tags"):
        # Get negation vector from FWL
        tokens, negation_vector = nh_baseline(text, negation_approach)

        # Tag negated tokens with NOT_ prefix
        tagged_tokens = []
        for token, neg_val in zip(tokens, negation_vector):
            if neg_val == 2:  # Token in negation scope
                tagged_token = f"NOT_{token}"
                tagged_tokens.append(tagged_token)
                not_word_count += 1
            else:  # Normal token (including cue with value 1)
                tagged_tokens.append(token)

        all_tagged_tokens.append(tagged_tokens)

    # Create word index from original Word2Vec + NOT_ variants
    word_index = {word: i+1 for i, word in enumerate(word2vec_model.wv.index_to_key)}

    # Add NOT_ words to vocabulary
    next_idx = len(word_index) + 1
    not_words_added = set()

    for tokens in all_tagged_tokens:
        for token in tokens:
            if token.startswith("NOT_") and token not in word_index:
                word_index[token] = next_idx
                next_idx += 1
                not_words_added.add(token)

    vocab_size = len(word_index) + 1  # +1 for OOV token

    # Convert tagged tokens to sequences
    all_sequences = []
    for tagged_tokens in all_tagged_tokens:
        sequence = []
        for token in tagged_tokens:
            if token in word_index:
                sequence.append(word_index[token])
            else:
                sequence.append(0)  # OOV token
        all_sequences.append(sequence)

    # Pad sequences
    X = pad_sequences(all_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    y = labels

    # Vocabulary statistics
    vocab_stats = {
        'original_vocab_size': len(word2vec_model.wv),
        'not_words_added': len(not_words_added),
        'total_vocab_size': vocab_size,
        'not_word_count': not_word_count
    }

    return X, y, word_index, vocab_stats

def create_embedding_matrix(word2vec_model, word_index):
    """
    Create embedding matrix from Word2Vec model (VECTOR-LEVEL)

    Args:
        word2vec_model: loaded Word2Vec model
        word_index: dictionary mapping words to indices

    Returns:
        embedding_matrix: matrix of word vectors
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIMENSIONS))

    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    return embedding_matrix

def create_embedding_matrix_word_level(word2vec_model, word_index):
    """
    Create embedding matrix with NOT_ words initialized as -original_word (WORD-LEVEL)

    Args:
        word2vec_model: loaded Word2Vec model
        word_index: dictionary mapping words to indices (including NOT_ words)

    Returns:
        embedding_matrix: matrix of word vectors with NOT_ = -original
    """
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIMENSIONS))

    not_words_found = 0
    not_words_missing = 0

    for word, i in word_index.items():
        if word.startswith("NOT_"):
            # Get original word
            original_word = word[4:]  # Remove "NOT_" prefix
            if original_word in word2vec_model.wv:
                # Initialize as NEGATIVE of original embedding
                embedding_matrix[i] = -word2vec_model.wv[original_word]
                not_words_found += 1
            else:
                # Original word not in vocabulary, stays as zero vector
                not_words_missing += 1
        else:
            # Normal word
            if word in word2vec_model.wv:
                embedding_matrix[i] = word2vec_model.wv[word]

    print(f"  NOT_ words initialized: {not_words_found}")
    print(f"  NOT_ words without original: {not_words_missing}")

    return embedding_matrix

def get_negation_model_word_level(vocab_size, embedding_matrix, input_length=MAX_SEQUENCE_LENGTH):
    """
    Create model for WORD-LEVEL negation (single input, no negation embedding)
    Uses NOT_ tagged tokens in vocabulary
    """
    # Single word input (includes NOT_ tokens)
    word_input = Input(shape=(input_length,), name='word_input')
    word_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIMENSIONS,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False,
        name='word_embedding'
    )(word_input)

    # BiLSTM layers (same architecture as vector-level)
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(word_embedding)
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(x)
    x = Conv1D(100, 5, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=word_input, outputs=output, name="Word_Level_Negation_Model")
    return model

def get_negation_model(vocab_size, embedding_matrix, input_length=MAX_SEQUENCE_LENGTH):
    if EMBEDDING_INTEGRATION_MODE == 0:
        return get_negation_model_concat(vocab_size, embedding_matrix, input_length)
    elif EMBEDDING_INTEGRATION_MODE == 1:
        return get_negation_model_feature_interact(vocab_size, embedding_matrix, input_length)
    elif EMBEDDING_INTEGRATION_MODE == 2:
        return get_negation_model_concat_plus_attention(vocab_size, embedding_matrix, input_length)
    elif EMBEDDING_INTEGRATION_MODE == 3:
        return get_negation_model_focused_attention(vocab_size, embedding_matrix, input_length)

# concat only
def get_negation_model_concat(vocab_size, embedding_matrix, input_length=MAX_SEQUENCE_LENGTH):
    # Word input and embedding
    word_input = Input(shape=(input_length,), name='word_input')
    word_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIMENSIONS,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False,
        name='word_embedding'
    )(word_input)

    # Negation input and embedding
    negation_input = Input(shape=(input_length,), name='negation_input')
    negation_embedding_matrix = np.random.normal(size=(4, NEGATION_EMBEDDING_DIMENSIONS))
    negation_embedding_matrix[0] = np.zeros(NEGATION_EMBEDDING_DIMENSIONS)  # zero vector for 'no negation'

    negation_embedding = Embedding(
        input_dim=4,
        output_dim=NEGATION_EMBEDDING_DIMENSIONS,
        weights=[negation_embedding_matrix],
        input_length=input_length,
        trainable=True,
        name='negation_embedding'
    )(negation_input)

    # Concatenate word and negation embeddings along the feature dimension
    # This is more suitable for BiLSTM architectures
    combined_embedding = Concatenate(axis=-1, name='concatenate_embeddings')(
        [word_embedding, negation_embedding]
    )

    # BiLSTM layers work well with concatenated features
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(combined_embedding)
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(x)
    x = Conv1D(100, 5, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[word_input, negation_input], outputs=output, name="Negation_Sentiment_Model")
    return model

# add attention mechanism
def get_negation_model_concat_plus_attention(vocab_size, embedding_matrix, input_length=MAX_SEQUENCE_LENGTH):
    """
    Create model with negation handling using concatenation approach with attention

    Args:
        vocab_size: size of vocabulary
        embedding_matrix: pre-trained embedding matrix
        input_length: length of input sequences

    Returns:
        model: Keras model
    """
    # Word input and embedding
    word_input = Input(shape=(input_length,), name='word_input')
    word_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIMENSIONS,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False,
        name='word_embedding'
    )(word_input)

    # Negation input and embedding
    negation_input = Input(shape=(input_length,), name='negation_input')
    if not ZERO_VECTOR:
        negation_embedding = Embedding(
            input_dim=4,  # 0, 1, 2, 3 (your negation categories)
            output_dim=NEGATION_EMBEDDING_DIMENSIONS,
            input_length=input_length,
            name='negation_embedding'
        )(negation_input)
    else:
        negation_embedding_matrix = np.random.normal(size=(4, NEGATION_EMBEDDING_DIMENSIONS))
        negation_embedding_matrix[0] = np.zeros(NEGATION_EMBEDDING_DIMENSIONS)  # zero vector for 'no negation'

        negation_embedding = Embedding(
            input_dim=4,
            output_dim=NEGATION_EMBEDDING_DIMENSIONS,
            weights=[negation_embedding_matrix],
            input_length=input_length,
            trainable=True,
            name='negation_embedding'
        )(negation_input)

    # Concatenate word and negation embeddings along the feature dimension
    combined_embedding = Concatenate(axis=-1, name='concatenate_embeddings')(
        [word_embedding, negation_embedding]
    )

    # BiLSTM layers
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(combined_embedding)
    lstm_out = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(x)

    # Add attention layer
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention_weights = Activation('softmax')(attention)
    attention_weights = RepeatVector(LSTM_UNITS * 2)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)

    # Apply attention weights to BiLSTM output
    weighted_lstm = Multiply()([lstm_out, attention_weights])

    # Continue with Conv1D and subsequent layers
    x = Conv1D(100, 5, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(weighted_lstm)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[word_input, negation_input], outputs=output, name="Negation_Attention_Sentiment_Model")
    return model

# feature interaction approach
def get_negation_model_feature_interact(vocab_size, embedding_matrix, input_length=MAX_SEQUENCE_LENGTH):
    """
    Create model with negation handling using feature interaction approach

    Args:
        vocab_size: size of vocabulary
        embedding_matrix: pre-trained embedding matrix
        input_length: length of input sequences

    Returns:
        model: Keras model
    """
    # Word input and embedding
    word_input = Input(shape=(input_length,), name='word_input')
    word_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIMENSIONS,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False,
        name='word_embedding'
    )(word_input)

    # Negation input and embedding
    negation_input = Input(shape=(input_length,), name='negation_input')
    if not ZERO_VECTOR:
        negation_embedding = Embedding(
            input_dim=4,  # 0, 1, 2, 3 (your negation categories)
            output_dim=NEGATION_EMBEDDING_DIMENSIONS,
            input_length=input_length,
            name='negation_embedding'
        )(negation_input)
    else:
        negation_embedding_matrix = np.random.normal(size=(4, NEGATION_EMBEDDING_DIMENSIONS))
        negation_embedding_matrix[0] = np.zeros(NEGATION_EMBEDDING_DIMENSIONS)  # zero vector for 'no negation'

        negation_embedding = Embedding(
            input_dim=4,
            output_dim=NEGATION_EMBEDDING_DIMENSIONS,
            weights=[negation_embedding_matrix],
            input_length=input_length,
            trainable=True,
            name='negation_embedding'
        )(negation_input)

    # Project negation embeddings to same dimension as word embeddings
    negation_projection = Dense(EMBEDDING_DIMENSIONS, activation='tanh', name='negation_projection')(negation_embedding)

    # Add word embeddings and negation projections for direct interaction
    word_embedding_modulated = Add(name='word_negation_interaction')([word_embedding, negation_projection])

    # BiLSTM layers
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(word_embedding_modulated)
    lstm_out = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(x)

    # Add attention layer
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention_weights = Activation('softmax')(attention)
    attention_weights = RepeatVector(LSTM_UNITS * 2)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)

    # Apply attention weights to BiLSTM output
    weighted_lstm = Multiply()([lstm_out, attention_weights])

    # Continue with Conv1D and subsequent layers
    x = Conv1D(100, 5, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(weighted_lstm)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[word_input, negation_input], outputs=output, name="Negation_Interaction_Model")
    return model

# focused attention
def get_negation_model_focused_attention(vocab_size, embedding_matrix, input_length=MAX_SEQUENCE_LENGTH):
    """
    Create model with negation handling using focused attention mechanism

    Args:
        vocab_size: size of vocabulary
        embedding_matrix: pre-trained embedding matrix
        input_length: length of input sequences

    Returns:
        model: Keras model
    """
    # Word input and embedding
    word_input = Input(shape=(input_length,), name='word_input')
    word_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIMENSIONS,
        weights=[embedding_matrix],
        input_length=input_length,
        trainable=False,
        name='word_embedding'
    )(word_input)

    # Negation input and embedding
    negation_input = Input(shape=(input_length,), name='negation_input')
    if not ZERO_VECTOR:
        negation_embedding = Embedding(
            input_dim=4,  # 0, 1, 2, 3 (your negation categories)
            output_dim=NEGATION_EMBEDDING_DIMENSIONS,
            input_length=input_length,
            name='negation_embedding'
        )(negation_input)
    else:
        negation_embedding_matrix = np.random.normal(size=(4, NEGATION_EMBEDDING_DIMENSIONS))
        negation_embedding_matrix[0] = np.zeros(NEGATION_EMBEDDING_DIMENSIONS)  # zero vector for 'no negation'

        negation_embedding = Embedding(
            input_dim=4,
            output_dim=NEGATION_EMBEDDING_DIMENSIONS,
            weights=[negation_embedding_matrix],
            input_length=input_length,
            trainable=True,
            name='negation_embedding'
        )(negation_input)

    # Generate negation-focused attention weights
    negation_attention = Dense(64, activation='relu', name='negation_attention_1')(negation_embedding)
    negation_attention = Dense(1, activation='sigmoid', name='negation_attention_2')(negation_attention)

    # Reshape to match word embedding dimensions for multiplication
    negation_attention = Reshape((input_length, 1), name='negation_attention_reshape')(negation_attention)

    # Apply negation attention directly to word embeddings
    word_embedding_weighted = Multiply(name='negation_weighted_words')([word_embedding, negation_attention])

    # BiLSTM layers
    x = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(word_embedding_weighted)
    lstm_out = Bidirectional(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE,
                          recurrent_regularizer=l2(L2_REGULARIZATION),
                          return_sequences=True))(x)

    # Standard sequence attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention_weights = Activation('softmax')(attention)
    attention_weights = RepeatVector(LSTM_UNITS * 2)(attention_weights)
    attention_weights = Permute([2, 1])(attention_weights)

    # Apply attention weights to BiLSTM output
    weighted_lstm = Multiply()([lstm_out, attention_weights])

    # Continue with Conv1D and subsequent layers
    x = Conv1D(100, 5, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(weighted_lstm)
    x = GlobalMaxPool1D()(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(L2_REGULARIZATION))(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[word_input, negation_input], outputs=output, name="Negation_Focused_Attention_Model")
    return model

def lr_scheduler(epoch, lr):
    """
    Learning rate scheduler function

    Args:
        epoch: current epoch
        lr: current learning rate

    Returns:
        new_lr: new learning rate
    """
    decay_rate = (FINAL_LEARNING_RATE / INITIAL_LEARNING_RATE) ** (1.0 / EPOCHS)
    new_lr = lr * decay_rate
    return new_lr

def train_model(model, X_train, y_train, X_val, y_val, model_name, negation_train=None, negation_val=None):
    # Compile model
    optimizer = Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        LearningRateScheduler(lr_scheduler),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_SAVE_PATH, f"{model_name}_best.h5"),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        TensorBoard(log_dir=os.path.join(RESULTS_SAVE_PATH, f"logs/{model_name}"))
    ]

    # Train the model
    if negation_train is not None and negation_val is not None:
        # Training with negation vectors
        history = model.fit(
            [X_train, negation_train], y_train,
            validation_data=([X_val, negation_val], y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Training without negation vectors
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

    # Save the final model
    model.save(os.path.join(MODEL_SAVE_PATH, f"{model_name}_final.h5"))

    return history

def evaluate_model(model, X_test, y_test, model_name, negation_test=None, test_df=None):
    """
    Evaluate the model on test data and save misclassified examples with detailed information

    Args:
        model: Keras model
        X_test: test data
        y_test: test labels
        model_name: model name for saving results
        negation_test: test negation vectors (optional)
        test_df: test dataframe with original texts

    Returns:
        results: dictionary with evaluation metrics
    """
    # Predict on test data and measure inference time
    inference_start = time.time()
    if negation_test is not None:
        y_pred_prob = model.predict([X_test, negation_test], verbose=1)
    else:
        y_pred_prob = model.predict(X_test, verbose=1)
    inference_time = time.time() - inference_start

    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Calculate metrics
    cr = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Save misclassified data with more information
    misclassified_indices = np.where(y_test != y_pred)[0]

    # Create dataframe for misclassified examples
    misclassified_data = {
        'original_text': test_df.iloc[misclassified_indices]['text'].values,
        'true_label': y_test[misclassified_indices],
        'predicted_label': y_pred[misclassified_indices]
    }

    # Add negation vector if available
    if negation_test is not None:
        # Convert negation vectors to strings for CSV storage
        misclassified_data['negation_vector'] = [
            str(negation_test[idx].tolist()) for idx in misclassified_indices
        ]

    # Add word vector (indices)
    misclassified_data['word_vector'] = [
        str(X_test[idx].tolist()) for idx in misclassified_indices
    ]

    # Save to CSV
    misclassified_df = pd.DataFrame(misclassified_data)
    misclassified_df.to_csv(os.path.join(RESULTS_SAVE_PATH, f"{model_name}_misclassified.csv"), index=False)

    # Print results
    print(f"\nModel: {model_name}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(cm)

    # Save results
    results = {
        'classification_report': cr,
        'confusion_matrix': cm,
        'accuracy': cr['accuracy'],
        'f1_score': cr['weighted avg']['f1-score'],
        'precision_0': cr['0']['precision'],
        'precision_1': cr['1']['precision'],
        'recall_0': cr['0']['recall'],
        'recall_1': cr['1']['recall'],
        'f1_0': cr['0']['f1-score'],
        'f1_1': cr['1']['f1-score'],
        'inference_time': inference_time
    }

    print(f"Inference time: {inference_time:.2f} seconds")

    # Save results to file
    with open(os.path.join(RESULTS_SAVE_PATH, f"{model_name}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    return results

def plot_training_history(history, model_name):
    """
    Plot training and validation accuracy and loss

    Args:
        history: training history
        model_name: model name for saving plots
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_name} - Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_name} - Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_SAVE_PATH, f"{model_name}_history.png"))
    plt.close()

def print_comparison_table(vector_results, word_results, vector_train_time, word_train_time,
                           vector_vocab_size, word_vocab_size, word_vocab_stats):
    """
    Print comparison table between vector-level and word-level approaches
    """
    print("\n" + "="*80)
    print("COMPARISON: Vector-Level vs Word-Level Negation Augmentation")
    print("="*80)

    print("\nTraining Configuration:")
    print(f"  Dataset: PRDECT-ID")
    print(f"  Negation Approach: FWL (window=2)")
    print(f"  Architecture: BiLSTM + Conv1D + GlobalMaxPool")

    print("\n" + "-"*80)
    print(f"{'Metric':<30} {'Vector-Level':>20} {'Word-Level':>20}")
    print("-"*80)

    # Performance metrics
    print(f"{'Accuracy':<30} {vector_results['accuracy']:>20.4f} {word_results['accuracy']:>20.4f}")
    print(f"{'F1 Score (weighted)':<30} {vector_results['f1_score']:>20.4f} {word_results['f1_score']:>20.4f}")
    print(f"{'Precision (class 0)':<30} {vector_results['precision_0']:>20.4f} {word_results['precision_0']:>20.4f}")
    print(f"{'Precision (class 1)':<30} {vector_results['precision_1']:>20.4f} {word_results['precision_1']:>20.4f}")
    print(f"{'Recall (class 0)':<30} {vector_results['recall_0']:>20.4f} {word_results['recall_0']:>20.4f}")
    print(f"{'Recall (class 1)':<30} {vector_results['recall_1']:>20.4f} {word_results['recall_1']:>20.4f}")
    print(f"{'F1 Score (class 0)':<30} {vector_results['f1_0']:>20.4f} {word_results['f1_0']:>20.4f}")
    print(f"{'F1 Score (class 1)':<30} {vector_results['f1_1']:>20.4f} {word_results['f1_1']:>20.4f}")

    print("-"*80)

    # Time metrics
    print(f"{'Training Time (min)':<30} {vector_train_time/60:>20.2f} {word_train_time/60:>20.2f}")
    print(f"{'Inference Time (sec)':<30} {vector_results['inference_time']:>20.2f} {word_results['inference_time']:>20.2f}")

    print("-"*80)

    # Architecture metrics
    print(f"{'Input Dimension':<30} {'216 (200+16)':>20} {'200 (word only)':>20}")
    print(f"{'Vocabulary Size':<30} {vector_vocab_size:>20,} {word_vocab_size:>20,}")

    if word_vocab_stats:
        print(f"{'NOT_ words added':<30} {'-':>20} {word_vocab_stats['not_words_added']:>20,}")
        print(f"{'NOT_ token count':<30} {'-':>20} {word_vocab_stats['not_word_count']:>20,}")

    print("="*80)

    # Save to CSV
    comparison_data = {
        'Metric': ['Accuracy', 'F1 Score', 'Precision 0', 'Precision 1', 'Recall 0', 'Recall 1',
                   'F1 Score 0', 'F1 Score 1', 'Training Time (min)', 'Inference Time (sec)',
                   'Vocabulary Size'],
        'Vector-Level': [
            vector_results['accuracy'], vector_results['f1_score'],
            vector_results['precision_0'], vector_results['precision_1'],
            vector_results['recall_0'], vector_results['recall_1'],
            vector_results['f1_0'], vector_results['f1_1'],
            vector_train_time/60, vector_results['inference_time'],
            vector_vocab_size
        ],
        'Word-Level': [
            word_results['accuracy'], word_results['f1_score'],
            word_results['precision_0'], word_results['precision_1'],
            word_results['recall_0'], word_results['recall_1'],
            word_results['f1_0'], word_results['f1_1'],
            word_train_time/60, word_results['inference_time'],
            word_vocab_size
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(RESULTS_SAVE_PATH, "comparison_vector_vs_word.csv"), index=False)
    print(f"\nComparison saved to: {RESULTS_SAVE_PATH}comparison_vector_vs_word.csv")

def main():
    overall_start_time = time.time()

    if TRAIN_BOTH:
        print("="*80)
        print("COMPARISON MODE: Training Both Vector-Level and Word-Level Models")
        print("="*80)
    else:
        mode = "Word-Level" if USE_WORD_LEVEL else "Vector-Level"
        print(f"Training {mode} Model Only")

    print("Dataset: PRDECT-ID")
    print("Negation Approach: FWL (window=2)")

    # Load data (shared)
    train_df, validation_df, test_df = load_data_partitioned()

    # Train Word2Vec model from scratch (shared)
    print("\nTraining Word2Vec model from scratch...")
    model_save_path = "models/word2vec_custom.model"
    word2vec_model = train_word2vec_model(pd.concat([train_df, validation_df, test_df], ignore_index=True), model_save_path)

    if TRAIN_BOTH or not USE_WORD_LEVEL:
        # ========================= VECTOR-LEVEL Model =========================
        print("\n" + "="*80)
        print("1. Training VECTOR-LEVEL Negation Model (concat embedding)")
        print("="*80)

        # Preprocess for vector-level
        print("\nPreprocessing data with vector-level approach...")
        word_index_vector = {word: i+1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
        vocab_size_vector = len(word_index_vector) + 1

        X_train_vec, y_train_vec, negation_train_vec = preprocess_data(
            train_df, word2vec_model, with_negation=True, negation_approach=5
        )
        X_val_vec, y_val_vec, negation_val_vec = preprocess_data(
            validation_df, word2vec_model, with_negation=True, negation_approach=5
        )
        X_test_vec, y_test_vec, negation_test_vec = preprocess_data(
            test_df, word2vec_model, with_negation=True, negation_approach=5
        )

        # Create embedding matrix
        embedding_matrix_vector = create_embedding_matrix(word2vec_model, word_index_vector)

        # Create model
        print("\nCreating vector-level model...")
        vector_model = get_negation_model(vocab_size_vector, embedding_matrix_vector)
        vector_model.summary()

        # Train
        print("\nTraining vector-level model...")
        vector_train_start = time.time()
        vector_history = train_model(
            vector_model, X_train_vec, y_train_vec, X_val_vec, y_val_vec,
            model_name="fwl_vector_level", negation_train=negation_train_vec, negation_val=negation_val_vec
        )
        vector_train_time = time.time() - vector_train_start

        # Evaluate
        print("\nEvaluating vector-level model...")
        vector_results = evaluate_model(
            vector_model, X_test_vec, y_test_vec,
            model_name="fwl_vector_level", negation_test=negation_test_vec, test_df=test_df
        )

        # Plot
        plot_training_history(vector_history, "fwl_vector_level")

        print(f"\nVector-Level Training Time: {vector_train_time/60:.2f} minutes")

    if TRAIN_BOTH or USE_WORD_LEVEL:
        # ========================= WORD-LEVEL Model =========================
        print("\n" + "="*80)
        print("2. Training WORD-LEVEL Negation Model (NOT_ tagging)")
        print("="*80)

        # Preprocess for word-level
        print("\nPreprocessing data with word-level approach (NOT_ tagging)...")
        X_train_word, y_train_word, word_index_word_train, vocab_stats_train = preprocess_data_word_level(
            train_df, word2vec_model, negation_approach=5
        )
        X_val_word, y_val_word, _, _ = preprocess_data_word_level(
            validation_df, word2vec_model, negation_approach=5
        )
        X_test_word, y_test_word, _, _ = preprocess_data_word_level(
            test_df, word2vec_model, negation_approach=5
        )

        # Use train vocabulary for consistency
        vocab_size_word = len(word_index_word_train) + 1

        print(f"\nVocabulary Statistics:")
        print(f"  Original vocab size: {vocab_stats_train['original_vocab_size']}")
        print(f"  NOT_ words added: {vocab_stats_train['not_words_added']}")
        print(f"  Total vocab size: {vocab_stats_train['total_vocab_size']}")
        print(f"  NOT_ token count in train: {vocab_stats_train['not_word_count']}")

        # Create embedding matrix with NOT_ = -original
        print("\nCreating embedding matrix with NOT_ words...")
        embedding_matrix_word = create_embedding_matrix_word_level(word2vec_model, word_index_word_train)

        # Create model
        print("\nCreating word-level model...")
        word_model = get_negation_model_word_level(vocab_size_word, embedding_matrix_word)
        word_model.summary()

        # Train
        print("\nTraining word-level model...")
        word_train_start = time.time()
        word_history = train_model(
            word_model, X_train_word, y_train_word, X_val_word, y_val_word,
            model_name="fwl_word_level", negation_train=None, negation_val=None
        )
        word_train_time = time.time() - word_train_start

        # Evaluate
        print("\nEvaluating word-level model...")
        word_results = evaluate_model(
            word_model, X_test_word, y_test_word,
            model_name="fwl_word_level", negation_test=None, test_df=test_df
        )

        # Plot
        plot_training_history(word_history, "fwl_word_level")

        print(f"\nWord-Level Training Time: {word_train_time/60:.2f} minutes")

    # ========================= Comparison =========================
    if TRAIN_BOTH:
        print_comparison_table(
            vector_results, word_results,
            vector_train_time, word_train_time,
            vocab_size_vector, vocab_size_word,
            vocab_stats_train
        )

    elapsed_time = time.time() - overall_start_time
    print(f"\n" + "="*80)
    print(f"Total execution time: {elapsed_time/60:.2f} minutes ({elapsed_time:.2f} seconds)")
    print("="*80)

if __name__ == "__main__":
    main()
