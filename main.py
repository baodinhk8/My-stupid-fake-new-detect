from flask_cors import CORS, cross_origin
from flask import Flask, request, render_template
from keras import initializers, regularizers, constraints, optimizers, layers
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Model
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from gensim.models import KeyedVectors
import gensim
import itertools
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from vncorenlp import VnCoreNLP
annotator = VnCoreNLP(address="http://127.0.0.1", port=9000)

linear_clf = PassiveAggressiveClassifier(n_iter_no_change=50)

# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()

NB = MultinomialNB()

clf = DecisionTreeClassifier()


#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()

df = pd.read_csv('data.csv')

unusual_chars = ["'", '"', "(", ")", ";", ":", ",", "<", ".", ">", "?", "/",
                 "{", "}", "[", "]", "|", "_", "-", "+", "=", "!", "@", "#", "$", "%", "^", "&", "*", "~", "`", "”", "", '']
new_wst = []


def tokenize(text):
    word_segmented_text = annotator.tokenize(text)

    for sentence in word_segmented_text:
        new_sentence = []
        for word in sentence:
            word = ''.join([i for i in word if not i.isdigit()])  # rm num
            boolen = 0
            for unusual_char in unusual_chars:
                if word is unusual_char:
                    boolen = 1
                    break
            if not(boolen):
                new_sentence.append(word.lower())
        sentence_wst = " ".join(new_sentence)
        # new_wst.append(sentence_wst)

    return(sentence_wst)


content_array = []
for content in df.content:
    tokenize_content = tokenize(content)
    content_array.append(tokenize_content)

y = df.label
# tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
# X=tfidf_v.fit_transform(content_array).toarray()
count_vectorizer = CountVectorizer()

count_vectorizer.fit_transform(content_array)

freq_term_matrix = count_vectorizer.transform(content_array)

tfidf = TfidfTransformer(norm="l2")
tfidf.fit(freq_term_matrix)
tf_idf_matrix = tfidf.fit_transform(freq_term_matrix)

print(tf_idf_matrix.shape)

#articals=["", ""]
articals = []
new_artical = []

while (1):
    print("***Press Enter too quit input***")
    art = input("Input artical: ")
    if(art == ""):
        break
    articals.append(art)


for artical in articals:
    tokenize_artical = tokenize(artical)
    new_artical.append(tokenize_artical)
print(new_artical)

freq_term_matrix_P = count_vectorizer.transform(new_artical)

tf_idf_matrix_P = tfidf.transform(freq_term_matrix_P)

print(tf_idf_matrix_P.shape)

X_test = tf_idf_matrix_P
y_test = [1, 0]

X_train = tf_idf_matrix
y_train = y

x_train = content_array
x_test = new_artical

embed_size = 300
max_feature = 50000
max_len = 2000
tokenizer = Tokenizer(num_words=max_feature)

tokenizer.fit_on_texts(x_train)

x_train_features = np.array(tokenizer.texts_to_sequences(x_train))
x_test_features = np.array(tokenizer.texts_to_sequences(x_test))

x_train_features = pad_sequences(x_train_features, maxlen=max_len)
x_test_features = pad_sequences(x_test_features, maxlen=max_len)

x_train_features

embed_size = 300

inp = Input(shape=(max_len,))
x = Embedding(max_feature, embed_size)(inp)
x = Bidirectional(GRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print(model.summary())

y_test = np.array(y_test)

# 35,40,200 is now the perfect epochs number
history = model.fit(x_train_features, y_train, batch_size=512,
                    epochs=40, validation_data=(x_test_features, y_test))

y_predict = [1 if o > 0.5 else 0 for o in model.predict(x_test_features)]
per = [o[0]*100 if o > 0.5 else (1-o[0]) *
       100 for o in model.predict(x_test_features)]
y_predict
y_test
for i in range(len(y_test)):
    if y_test[i] == 1:
        print(articals[i][:30]+"... : " + str(per[i]) + "% là bài viết thật")
    else:
        print(articals[i][:30]+"... : " + str(per[i]) + "% là bài viết giả")


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/")
@cross_origin()
def main():
    return render_template("index.html")


@app.route("/", methods=['POST'])
@cross_origin()
def index_main():
    ct = []
    ct.append(request.form.get('input'))

    x_test_features = np.array(tokenizer.texts_to_sequences(ct))

    x_test_features = pad_sequences(x_test_features, maxlen=max_len)

    for o in model.predict(x_test_features):
        if o[0] > 0.5:
            return render_template("index.html", result=(str(o[0]*100)+"% là bài viết thật"))
        else:
            return render_template("index.html", result=(str(100-o[0]*100)+"% là bài viết giả"))


@app.route("/test")
@cross_origin()
def test():
    return "my super smart API work well ;)"


@app.route("/predict", methods=['POST'])
@cross_origin()
def index():
    ct = []
    content = request.json
    print(content)
    ct.append(content["artical"])

    x_test_features = np.array(tokenizer.texts_to_sequences(ct))

    x_test_features = pad_sequences(x_test_features, maxlen=max_len)

    for o in model.predict(x_test_features):
        return str(o[0])


if __name__ == '__main__':
    app.run()
