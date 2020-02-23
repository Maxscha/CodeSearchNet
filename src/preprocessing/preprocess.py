from .ngram import make_to_ngram
import nltk
# Load library
from nltk.corpus import stopwords

# You will have to download the set of stop words the first time
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

def preprocess(text):
    sno = nltk.stem.SnowballStemmer('english')
    stop_words = set(stopwords.words('english'))
    wordnet_lemmatizer = WordNetLemmatizer()

    text = list([wordnet_lemmatizer.lemmatize(x) for x in text if x not in stop_words])
    # return make_to_ngram(text, 3)
    return text