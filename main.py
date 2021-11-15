import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
def preprocess_data(data):
    '''
    Preprocess general data:
    Removing duplicate value and one hot encoding

    '''
    # check jika ada data yang terduplikasi (ada 10 data)
    data = data.drop_duplicates(subset='title', keep=False)
    data = data.drop(columns=['label'])

    return data

def custom_query(text):
    '''
    Predict custom query
    '''
    vectorizer = TfidfVectorizer()
    text = vectorizer.fit_transform([text]).toarray()
    return text
    
    



def main():
    DATASET_DIR = '/Users/teguhsatya/Projects/CLICKID/dataset/archive/annotated/combined/csv/all_agree.csv'
    df = pd.read_csv(DATASET_DIR)
    df_clean = preprocess_data(df)
    # splitting the data
    X = df_clean['title']
    y = df_clean['label_score']
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    ml = MultinomialNB()
    ml.fit(X_train, y_train)
    y_pred = ml.predict(X_test)
    
    target_names = ['Clickbait', 'Non-Clickbait'] # 85%
    print(classification_report(y_test, y_pred, target_names=target_names))
    


    
   
    

if __name__ == '__main__':
    main()
