from .ngram import make_to_ngram
def preprocess(text):
    return make_to_ngram(text, 3)