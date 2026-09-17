# Transformer Neural Network Architecture

A small, simple transformer implementation for natural langage processing implemented from scratch in PyTorch. Can be trained using simple datasets such as WikiText.txt or Shakespeare.txt for word/character level next token prediction.

## Features
- Multi-head self attention implementation using PyTorches scaled_dot_product_attention as a similarity metric.
- Fused query, key and value matrices for large matrix multiplication speedup.
- RMSNorm for stable normalization.
- Optionally tired input/output embedding weights for lower memory footprint.
- Next token prediction using pythonic generator objects and controllable temperature scaling.

## Requirements 

```
torch
tqdm
```

## Usage

```python
from transformer import Transformer, train_transformer
from text_dataset import TextDataset

dataset = TextDataset('wikitext.txt', seqLength=256, stride=128, minTokenFreq=6)

model = Transformer(dataset.vocabSize(), dim=256, blocks=6, heads=8, mlpDim=1024)

train_transformer(model, dataset, bsize=128, lr=3e-4, epochs=20)

for token in model.generate("hello ", dataset, maxLength=100, temperature=0.5):
    print(token, end="")

```


## Licence

MIT