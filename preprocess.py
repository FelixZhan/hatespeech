import datasets
import pandas as pd
import re
from cleantext import clean

def clean_text(text):
    cleaned_text = clean(text, no_emoji=True)
    cleaned_text = re.sub(r'[^a-zA-Z0-9 .?!#]', r'', cleaned_text)
    return cleaned_text

def load_and_clean_english_csv(filepath):
    df_english = pd.read_csv(filepath)
    df_english['text'] = df_english['text'].apply(clean_text)
    df_english['class'] = [1 if i == 'Hateful' else 0 for i in df_english['class']]
    df_english.rename(columns={'class': 'hatespeech'}, inplace=True)
    return df_english
def main():
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')
    df_hate_speech = dataset['train'].to_pandas()
    df_hate_speech.describe()
    df_hate_speech['hatespeech'] = [1 if i > 0 else 0 for i in df_hate_speech['hatespeech']]
    df_hate_speech.drop([i for i in df_hate_speech.columns if i not in ['text', 'hatespeech']], inplace=True, axis=1)
    df_hate_speech['text'] = df_hate_speech['text'].apply(clean_text)

    english_path = r"C:\Users\felix\Documents\Stanford\2023-24\DATASCI 112\final project"
    df_english = load_and_clean_english_csv(english_path + "\english.csv")

    df_concatenated = pd.concat([df_hate_speech, df_english], ignore_index=True)
    for i in [0.10, 0.15, 0.25]:
      df_reduced = df_concatenated.sample(frac=i, random_state=42)  # to make data faster
      df_reduced.to_csv(english_path + f"\combined_english_{i}.csv")

if __name__ == "__main__":
    main()