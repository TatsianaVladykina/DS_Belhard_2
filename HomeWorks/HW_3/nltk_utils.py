import json
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Разбиваем предложение на массив слов/токенов
    Токен может быть словом, знаком препинания или числом
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Стемминг = находим корневую форму слова
    Примеры:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Возвращаем мешок слов в виде массива:
    1 для каждого известного слова, которое существует в предложении, 0 в противном случае
    Пример:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # Стемминг каждого слова
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Инициализация мешка нулями для каждого слова
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag