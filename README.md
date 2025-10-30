# Comparing Vector-Level vs Word-Level Negation Augmentation

This folder contains an implementation comparing two approaches for incorporating negation information in sentiment analysis:
1. **Vector-Level Augmentation** - Negation as separate embedding features
2. **Word-Level Augmentation** - NOT_ prefix tagging on negated tokens

## Overview

- **Dataset**: PRDECT-ID only
- **Negation Approach**: FWL (Fixed Window Length) with window size = 2
- **Architecture**: BiLSTM + Conv1D + GlobalMaxPool
- **Comparison**: Performance, training time, inference time, vocabulary size

## Two Approaches Compared

### 1. Vector-Level Augmentation (Current Standard)
- Negation info as separate embedding (16 dimensions)
- Concatenated with word embedding (200 dimensions)
- Total input: 216 dimensions per token
- Dual input model: `[word_input, negation_input]`

**Architecture:**
```
Word Embedding (200 dim) ─┐
                          ├─ Concat (216 dim) ─> BiLSTM ─> Conv1D ─> Output
Negation Embedding (16)  ─┘
```

### 2. Word-Level Augmentation (Novel Experiment)
- Negation info via NOT_ prefix on tokens
- Single word embedding (200 dimensions)
- NOT_word initialized as `-word_embedding`
- Single input model: `[word_input]`

**Architecture:**
```
Word Embedding (200 dim, includes NOT_ words) ─> BiLSTM ─> Conv1D ─> Output
```

**Example:**
```
Input           : "saya tidak suka makan nasi"
Negation Vector : [0, 1, 2, 2, 0]                       // vector level
Negation Tag    : "saya tidak NOT_suka NOT_makan nasi"  // word level
```

## Usage

### Train Both Models (Comparison Mode)
```bash
cd puf-02
python main_bilstm_fwl.py
```

Default setting `TRAIN_BOTH = True` will:
1. Train vector-level model
2. Train word-level model
3. Display comparison table
4. Save comparison CSV

### Train Single Model
Edit `main_bilstm_fwl.py`:
```python
TRAIN_BOTH = False
USE_WORD_LEVEL = True   # True for word-level, False for vector-level
```

## Configuration

```python
# Comparison Mode
TRAIN_BOTH = True          # Train both models for comparison
USE_WORD_LEVEL = False     # Only used if TRAIN_BOTH = False

# Dataset & Hyperparameters
DATASET_ID = 0             # PRDECT-ID
MAX_SEQUENCE_LENGTH = 185
BATCH_SIZE = 32
EPOCHS = 20
EMBEDDING_DIMENSIONS = 200
NEGATION_EMBEDDING_DIMENSIONS = 16  # Only for vector-level
LSTM_UNITS = 128
DROPOUT_RATE = 0.5
```

## Comparison Metrics

The script compares:

### Performance Metrics
- Accuracy
- F1 Score (weighted)
- Precision (class 0 & 1)
- Recall (class 0 & 1)
- F1 Score (class 0 & 1)

### Efficiency Metrics
- Training Time (minutes)
- Inference Time (seconds)
- Model Parameters
- Vocabulary Size
- NOT_ words statistics (word-level only)

## Outputs

After training, you'll find:

### Models (`models/`)
**Vector-Level:**
- `fwl_vector_level_best.h5`
- `fwl_vector_level_final.h5`

**Word-Level:**
- `fwl_word_level_best.h5`
- `fwl_word_level_final.h5`

**Shared:**
- `word2vec_custom.model`

### Results (`results/`)
**Vector-Level:**
- `fwl_vector_level_history.png`
- `fwl_vector_level_results.pkl`
- `fwl_vector_level_misclassified.csv`

**Word-Level:**
- `fwl_word_level_history.png`
- `fwl_word_level_results.pkl`
- `fwl_word_level_misclassified.csv`

**Comparison:**
- `comparison_vector_vs_word.csv`

### Logs (`results/logs/`)
- `fwl_vector_level/` - TensorBoard logs
- `fwl_word_level/` - TensorBoard logs

## Actual Comparison Results

```
================================================================================
COMPARISON: Vector-Level vs Word-Level Negation Augmentation
================================================================================

Metric                         Vector-Level           Word-Level
--------------------------------------------------------------------------------
Accuracy                             0.8838               0.8999
F1 Score (weighted)                  0.8836               0.8999
Precision (class 0)                  0.8730               0.9014
Precision (class 1)                  0.8967               0.8982
Recall (class 0)                     0.9102               0.9078
Recall (class 1)                     0.8549               0.8912
F1 Score (class 0)                   0.8912               0.9046
F1 Score (class 1)                   0.8753               0.8947
--------------------------------------------------------------------------------
Training Time (min)                   4.14                 3.93
Inference Time (sec)                  2.01                 2.09
--------------------------------------------------------------------------------
Input Dimension                 216 (200+16)        200 (word only)
Vocabulary Size                       4,199                5,940
NOT_ words added                          -                1,741
================================================================================
```

## Key Insights

### Word-Level Advantages (Better Performance):
- **Higher accuracy**: 89.99% vs 88.38%
- **Better F1 scores** across both classes
- Simpler architecture (fewer parameters)
- Faster training time (3.93 vs 4.14 minutes)
- More interpretable (vocabulary-based)
- Semantically meaningful (NOT_word = -word)
- Smaller vocabulary size despite adding NOT_ words

### Vector-Level Characteristics:
- Explicit negation representation as separate features
- Can distinguish negation cue vs scope
- More flexible learning potential
- Better recall for class 0 (91.02%)
- Slightly slower but comparable inference time

## Requirements

```bash
pip install tensorflow flair gensim scikit-learn pandas numpy tqdm matplotlib
```

## Notes

- POS tagger: `../resources/taggers/example-universal-pos/best-model.pt`
- GPU acceleration enabled (CUDA device 0)
- Word2Vec trained from scratch
- Early stopping with patience=5
- NOT_word embeddings initialized as negative of original word

## Troubleshooting

### GPU Issues
Edit `NegationHandlingBaseline.py` line 16:
```python
flair.device = torch.device("cpu")  # Change from cuda:0
```

### Memory Issues
Reduce hyperparameters:
```python
BATCH_SIZE = 16
LSTM_UNITS = 64
EMBEDDING_DIMENSIONS = 100
```

## Research Purpose

This implementation serves as a "side research" to investigate:
1. Which level of feature augmentation is more effective?
2. Trade-off between model complexity and performance
3. Computational efficiency gains
4. Interpretability benefits

### Main Findings:
- **Word-Level augmentation outperforms Vector-Level** (89.99% vs 88.38% accuracy)
- Simpler architecture does not sacrifice performance
- Training efficiency is comparable between both approaches
- Word-level approach offers better interpretability with higher accuracy

Results suggest that word-level negation handling (NOT_ prefix) is a promising alternative to traditional vector-level augmentation for sentiment analysis tasks.



