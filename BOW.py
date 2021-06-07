import nltk
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# If not previously performed:
# nltk.download('stopwords')

stemming = PorterStemmer()
stops = set(stopwords.words("english"))
def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X


def clean_text(raw_text):
    """This function works on a raw text string, and:
        1) changes to lower case
        2) tokenizes (breaks down into words
        3) removes punctuation and non-word text
        4) finds word stems
        5) removes stop words
        6) rejoins meaningful stem words"""
    
    # Convert to lower case
    text = raw_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # Stemming
    stemmed_words = [stemming.stem(w) for w in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in stemmed_words if not w in stops]
    
    # Rejoin meaningful stemmed words
    joined_words = ( " ".join(meaningful_words))
    
    # Return cleaned data
    return joined_words



def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer
    
    print ('Creating bag of words...')
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    
    # In this example features may be single words or two consecutive words
    # (as shown by ngram_range = 1,2)
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (1,2), \
                                 max_features = 10000
                                ) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings. The output is a sparse array
    train_data_features = vectorizer.fit_transform(X)
    
    # Convert to a NumPy array for easy of handling
    train_data_features = train_data_features.toarray()
    
    # tfidf transform
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()

    # Get words in the vocabulary
    vocab = vectorizer.get_feature_names()
   
    return vectorizer, vocab, train_data_features, tfidf_features, tfidf



datas=pd.read_csv('Clubhouse.csv')
dataf=list(datas['text'])
y=list(datas['Label'])
y = np.array(y).astype(int)
print(dataf[420])

# Get text to clean
text_to_clean = dataf
# Clean text
cleaned_text = apply_cleaning_function_to_list(text_to_clean)

vectorizer, vocab, data_features, tfidf_features, tfidf  = \
    create_bag_of_words(dataf)

X=data_features
kf=KFold(n_splits=10, random_state=None, shuffle=True)

acc=[]
prec=[]
reca=[]
f=[]


for train_index, test_index in kf.split(X):
        
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print(y_train)
        clf= LogisticRegression(C = 100,random_state = 0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict( X_test)
        acc.append(accuracy_score(y_test,  y_pred))
        reca.append(recall_score(y_test,  y_pred))
        prec.append(precision_score(y_test,  y_pred))
        f.append(f1_score(y_test,  y_pred))
        
        
print("accuracy : "+ str(np.mean(acc)))
print("recall : "+ str(np.mean(reca)))
print("precision : "+ str(np.mean(prec)))
print("f1 score : " + str(np.mean(f)))



print(data_features[10])