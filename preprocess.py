import datasets
import pandas as pd
import re
from cleantext import clean
dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
df = dataset['train'].to_pandas()
df.describe()
df['hatespeech'] = [1 if i > 0 else 0 for i in df['hatespeech']]
df.drop([i for i in df.columns if i not in ['text', 'hatespeech']], inplace=True, axis=1)

for line in df["text"]:
  clean(line, no_emoji=True)
  re.sub(r'[^a-zA-Z0-9 .?!#]',r'', line)