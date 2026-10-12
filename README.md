# Character-Level GPT using PyTorch

This is one of my learning projects where I implemented a small GPT-style language model from scratch using PyTorch.

I wanted to understand how Transformers work internally instead of just using a pretrained model. For this project, I used the Tiny Shakespeare dataset and trained the model to predict the next character based on the previous characters.

## What I Learned

While building this project, I explored:

- Character-level tokenization
- Token embeddings and positional embeddings
- Self-attention and query, key, and value
- Masked multi-head self-attention
- Feed-forward neural networks
- Residual connections and layer normalization
- Cross-entropy loss
- AdamW optimizer
- Training and validation loss
- Text generation using a trained model

## Model Architecture

I implemented a decoder-only Transformer with 6 Transformer blocks. Each block contains masked multi-head self-attention and a feed-forward network, along with residual connections and layer normalization.

The model predicts one character at a time and uses that prediction to generate the next character.

## Dataset

I used the Tiny Shakespeare dataset from Andrej Karpathy's GitHub repository.

Dataset: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

I converted the text into character indices and split it into 90% training data and 10% validation data.

## Training Configuration

- Framework: PyTorch
- Batch size: 64
- Context length: 256
- Embedding dimension: 48
- Attention heads: 6
- Transformer blocks: 6
- Learning rate: 0.0003
- Optimizer: AdamW
- Training iterations: 15,000
- Model size: Approximately 187K parameters

## Training Results

After training for 15,000 iterations, I got the following results:

| Step | Training Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1613 | 4.1637 |
| 5,000 | 1.5462 | 1.7252 |
| 10,000 | 1.4589 | 1.6446 |
| 15,000 | 1.4212 | 1.6007 |

Both training and validation loss decreased during training, showing that the model learned patterns from the dataset.

## Text Generation

The model generates Shakespeare-style text, including character names, punctuation, and dialogue-like formatting. However, the generated sentences are not always meaningful, and many words are misspelled.

Since this is a small character-level model, I still have a lot to learn about improving text generation.
