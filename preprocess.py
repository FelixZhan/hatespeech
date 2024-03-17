# -*- coding: utf-8 -*-
"""Final Project

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/164lBfYYCwKKH0l5i0NPnVfRm0bfnAdjV
"""

!pip install datasets

import datasets
import pandas as pd
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
df = dataset['train'].to_pandas()
df.describe()

df['hatespeech'] = [1 if i > 0 else 0 for i in df['hatespeech']]
df.drop([i for i in df.columns if i not in ['text', 'hatespeech']], inplace=True, axis=1)
df

!pip install clean-text

# todo: remove emojis, remove mentions, remove non-english,
import re
from cleantext import clean
for line in df["text"]:
  clean(line, no_emoji=True)
  re.sub(r'[^a-zA-Z0-9 .?!#]',r'', line)