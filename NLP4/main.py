import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.pipeline import Pipeline
import pymorphy2

df = pd.read_csv('labeled.csv', encoding="utf-8")

df['toxic'] = df['toxic'].apply(lambda x: 'Toxic' if x==1.0 else 'Not Toxic')


def clean_df(df):
    df = df.str.lower()
    df = df.replace(r'[^а-яА-Я]', ' ', regex=True)
    df = df.str.strip()
    df = df.apply(lambda x:' '.join([word for word in x.split() if word not in stopwords.words('russian') and len(word)>2]))
    return df

df['comment'] = clean_df(df['comment'])


df = df.dropna(subset=['comment'])
morph = pymorphy2.MorphAnalyzer()

def lem_tok(text):
    text_lem = [morph.parse(word)[0].normal_form for word in text.split(' ')]
    return text_lem


logreg = Pipeline([
                ('tfidf', TfidfVectorizer(sublinear_tf=True,
                        ngram_range=(1, 3),
                        tokenizer=lem_tok, max_df=1000)),
                ('clf', LinearSVC()),
])

logreg.fit(df['comment'], df['toxic'])


from flask import Flask, jsonify, request
from jsonrpc.backend.flask import api
app = Flask(__name__)
app.add_url_rule('/', 'api', api.as_view(), methods=['POST'])

@api.dispatcher.add_method
def classify(text):
  return logreg.predict([text])[0]


if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=9000)
