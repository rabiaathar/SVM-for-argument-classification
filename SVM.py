# -*- coding: utf-8 -*-
"""
"""
import pandas as pd
import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--predict_path', type=str)
    args = parser.parse_args()
    
    #reading bio data
    df = pd.read_table(args.train_path, delim_whitespace=True, names=('tokens', 'labels'))
    dfTest = pd.read_table(args.test_path, delim_whitespace=True, names=('tokens', 'labels'))
    
    #removing NaN values if any
    dfnew = df.replace(np.nan,'O')
    dfTestnew=dfTest.replace(np.nan,'O')
    
    train_tokens=dfnew['tokens']
    train_labels=dfnew['labels']
    
    test_tokens=dfTestnew['tokens']
    test_labels=dfTestnew['labels']
    
    #label encoding and decoding
    Encoder = LabelEncoder()
    train_labels = Encoder.fit_transform(train_labels)
    test_labels = Encoder.fit_transform(test_labels)
    labelMapping = dict(zip(Encoder.transform(Encoder.classes_),Encoder.classes_))
    labelMapping = dict(zip(Encoder.transform(Encoder.classes_),Encoder.classes_))
    
    #use tfidf as feature
    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(train_tokens)
    train_tfidf = Tfidf_vect.transform(train_tokens)
    test_tfidf = Tfidf_vect.transform(test_tokens)

    #pass tfidf to svm for training
    SVM = svm.SVC(C=0.5, kernel='linear', degree=3, gamma='auto')
    print('Training model..')
    SVM.fit(train_tfidf,train_labels)
    print('Predicting test data...')
    #making prediction on test data
    predictions = SVM.predict(test_tfidf)
    print('Writing predictions to File...')
    
    print('predictions are',len(predictions))
    print('data is ',len(test_tokens))
    #writting predictions data in text file
    with open(args.predict_path, 'a') as predict:
        for i in range(len(predictions)):
            predict.write(test_tokens[i])
            predict.write('\t')
            predict.write(labelMapping[predictions[i]])
            predict.write('\n')
         
        print('Predictions has been written to file.')   