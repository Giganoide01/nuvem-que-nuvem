from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

def load_model():
    global model
    # model variable refers to the global variable
    path = 'LSTM_model.h5'
    model = tf.keras.models.load_model(path)

def tokenizer(avaliacao):

    avaliacao = [avaliacao]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(avaliacao)
    encoded_docs = tokenizer.texts_to_sequences(avaliacao)
    tokenized_sent = pad_sequences(encoded_docs, maxlen=100)

    return tokenized_sent


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/shortenurl')
def shortenurl():
    shortcode=request.args['shortcode']
    data = tokenizer(shortcode)  # converts string tokenized
    prediction = model.predict(data)  # runs globally loaded model on the data
    if prediction > 0.5:
        veredito = 'Positiva'
    else:
        veredito = 'Negativa'
    return render_template('shortenurl.html', shortcode=veredito) 
 
if __name__ == '__main__':
    load_model()
    app.run(host='127.0.0.1', port=5000)
