# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:05:16 2020

@author: z003rukn
"""

import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import cohen_kappa_score

import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV


# data set
data = pd.read_csv("testdatalubeoil - corrected_extended.csv", sep=";", engine="python")

x = data.Sample
y = data.Classification


# Stopwords & Punctuation
my_stops = pd.read_csv("Stopwords.csv", sep=";", engine="python")
my_stops_list = my_stops.Stopwords.values.tolist()
punct = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~•’”“–°"


# Tokenizer
def tokenize(text):
    text = "".join([ch for ch in text if ch not in punct])
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        if item not in my_stops_list:
            stems.append(PorterStemmer().stem(item))
        else:
            stems.append(item)
    return stems

# without stemming
def tokenize_nostemm(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in punct])
    tokens = word_tokenize(text)
    return tokens

# -----------------------------------------------------------------------------
    
# Precision Recall Threshold Graphic
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 8))
    plt.title("Precision and recall scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision threshold")
    plt.legend(loc='best')
    #plt.yticks(np.arange(0, 1.1, step=0.1))
    #plt.xticks(np.arange(0, 1.1, step=0.1))


# -----------------------------------------------------------------------------
# TFIDF Vectorizer
## TODO: test different settings of vectorizer


# ,ngram_range=(1, 4)
# current vectorizer
vectorizer = TfidfVectorizer(tokenizer=tokenize,min_df=2,ngram_range=(1, 4),stop_words=my_stops_list)

# -----------------------------------------------------------------------------
 
# D2V Vectorizer
model = Doc2Vec.load("d2v_window3.model")

def vec_for_learning(doc2vec_model, tagged_docs):
    sents = tagged_docs
    targets, regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

def tag_docs(_data):
    tagged_data = [TaggedDocument(words=tokenize(_d), tags=[str(i)]) for i, _d in enumerate(_data)]
    return tagged_data



# -----------------------------------------------------------------------------

## TODO: print list of full vocabulary and sort by term frequency to analyze how many features are removed by min_df and validate if relevant information included

vectorizer_select="D2V"

iter_num = 1

for z in range(iter_num):

    
    xtrain, xtest, ytrain, ytest = train_test_split(data.Sample, data.Classification, stratify=data.Classification, test_size=0.3, random_state=0)
    
    if vectorizer_select == "TFIDF":
        xtrainvec = vectorizer.fit_transform(xtrain)
        xtestvec = vectorizer.transform(xtest)
        ytrain = ytrain.astype('int')
        ytest = ytest.astype('int')
        
        
        # Machine Learning Algorithm
        
        nb = MultinomialNB()
        nb.fit(xtrainvec, ytrain)
        
        svm = SVC(kernel="linear",random_state=0,probability=True)
        svm.fit(xtrainvec, ytrain)
        
        rf = RandomForestClassifier(bootstrap=False, random_state=0)
        rf.fit(xtrainvec, ytrain)
        
        logreg = LogisticRegression(random_state=0)
        logreg.fit(xtrainvec, ytrain)
        
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(xtrainvec, ytrain)
        
    
    
    elif vectorizer_select == "D2V":

        # hängt paragraph ID dran
        xtrain_D2V_tagged = tag_docs(xtrain)
        xtest_D2V_tagged = tag_docs(xtest)
        
        xlabel, xtrainvec = vec_for_learning(model, xtrain_D2V_tagged)
        ylabel, xtestvec = vec_for_learning(model, xtest_D2V_tagged)
        ytrain = ytrain.astype('int')
        ytest = ytest.astype('int')
        
        
        # Machine Learning Algorithm
        
        svm = SVC(kernel="linear",random_state=0,probability=True)
        svm.fit(xtrainvec, ytrain)
        
        rf = RandomForestClassifier(bootstrap=False, random_state=0)
        rf.fit(xtrainvec, ytrain)
        
        logreg = LogisticRegression(random_state=0)
        logreg.fit(xtrainvec, ytrain)
        
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(xtrainvec, ytrain)


# -----------------------------------------------------------------------------
    
# to control the features
    #feature_names = vectorizer.get_feature_names()
    #print("Feature names: ")
    #print(feature_names)
    #print("Number of features: ")
    #print(len(feature_names))
    #pd_feature_names = pd.DataFrame(feature_names)
    #print(pd_feature_names)
    #pd_feature_names.to_excel("C:/Users/z003rukn/Desktop/MA_Results/features_ngram1_4_stopwords", engine="xlsxwriter")

# -----------------------------------------------------------------------------    

    
# Results of Test Data
    
# Results Naive Bayes
    use_nb = False
    
    if use_nb:
        
        # Confusion Matrix, Classification Report, Accuracy
        y_pred_nb = nb.predict(xtestvec)
        
        confusion_nb = confusion_matrix(ytest, y_pred_nb)
        #print("Confusion matrix - Naive Bayes:")
        #print(confusion_nb)
        #skplt.metrics.plot_confusion_matrix(ytest, y_pred_nb,text_fontsize="large")
        #plt.savefig('Confusion Matrix NB.png')
        
        class_report_nb = classification_report(ytest,y_pred_nb)
        print("Classification report - Naive Bayes:")
        print(class_report_nb)
        
        acc_nb = accuracy_score(ytest, y_pred_nb)
        #print("Accuracy - Naive Bayes:")
        #print(acc_nb)  
        
        # Precision / Recall / Threshold     
        #y_scores_nb = nb.predict_proba(xtestvec)[:,1]
        #print(y_scores_nb)
        #p_nb, r_nb, thresholds_nb = precision_recall_curve(ytest, y_scores_nb)
        #plot_precision_recall_vs_threshold(p_nb,r_nb,thresholds_nb)
        #plt.savefig('prthreshold_NBnew.png')


# Results Support Vector Machine
    use_svm = True
    
    if use_svm:
        
        y_pred_svm = svm.predict(xtestvec)
        
        confusion_svm = confusion_matrix(ytest, y_pred_svm)
        #print("Confusion matrix - Support Vector Machine:")
        #print(confusion_svm)
        #skplt.metrics.plot_confusion_matrix(ytest, y_pred_svm,text_fontsize="large")
        #plt.savefig('Confusion Matrix SVM D2V.png')
    
        class_report_svm = classification_report(ytest,y_pred_svm)
        #print("Classification report - Support Vector Machine:")
        print(class_report_svm)
    
        acc_svm = accuracy_score(ytest, y_pred_svm)
        #print("Accuracy - Support Vector Machine:")
        #print(acc_svm)
    
        # Precision / Recall / Threshold     
        y_scores_svm = svm.predict_proba(xtestvec)[:,1]
        #print(y_scores_svm)
        p_svm, r_svm, thresholds_svm = precision_recall_curve(ytest, y_scores_svm)
        #plot_precision_recall_vs_threshold(p_svm,r_svm,thresholds_svm)
        #plt.savefig('prthreshold_SVMnewD2V.png')

   
# Results Random Forest
    use_rf = False
    
    if use_rf:
        
        y_pred_rf = rf.predict(xtestvec)
        
        confusion_rf = confusion_matrix(ytest, y_pred_rf)
        print("Confusion matrix - Random Forest:")
        print(confusion_rf)
        #skplt.metrics.plot_confusion_matrix(ytest, y_pred_rf,text_fontsize="large")
        #plt.savefig('Confusion Matrix D2V RFnew.png')
        
        class_report_rf = classification_report(ytest,y_pred_rf)
        print("Classification report - Random Forest:")
        print(class_report_rf)
        
        acc_rf = accuracy_score(ytest, y_pred_rf)
        #print("Accuracy - Random Forest:")
        #print(acc_rf)
    
    # Precision / Recall / Threshold     
        y_scores_rf = rf.predict_proba(xtestvec)[:,1]
        #print(y_scores_rf)
        p_rf, r_rf, thresholds_rf = precision_recall_curve(ytest, y_scores_rf)
        #plot_precision_recall_vs_threshold(p_rf,r_rf,thresholds_rf)
        #plt.savefig('prthreshold_RFnew D2V.png')

    
# Results Logistic Regression
    
    use_logreg = False
    
    if use_logreg:
        
        y_pred_logreg = logreg.predict(xtestvec)
        
        confusion_logreg = confusion_matrix(ytest, y_pred_logreg)
        print("Confusion matrix - Logistic Regression:")
        print(confusion_logreg)
        skplt.metrics.plot_confusion_matrix(ytest, y_pred_logreg,text_fontsize="large")
        plt.savefig('Confusion Matrix LogReg D2V NEW.png')
        
        class_report_logreg = classification_report(ytest,y_pred_logreg)
        print("Classification report - Logistic Regression:")
        print(class_report_logreg)
        
        acc_logreg = accuracy_score(ytest, y_pred_logreg)
        #print("Accuracy - Logistic Regression:")
        #print(acc_logreg)
        
        # Precision / Recall / Threshold     
        y_scores_logreg = logreg.predict_proba(xtestvec)[:,1]
        #print(y_scores_logreg)
        p_logreg, r_logreg, thresholds_logreg = precision_recall_curve(ytest, y_scores_logreg)
        plot_precision_recall_vs_threshold(p_logreg,r_logreg,thresholds_logreg)
        plt.savefig('prthreshold_LogReg D2V NEW.png')
        
        
# Results Decision Tree
        
    use_dt = False
    
    if use_dt:
        
        y_pred_dt = dt.predict(xtestvec)
        confusion_dt = confusion_matrix(ytest, y_pred_dt)
        print("Confusion matrix - Decision Tree:")
        print(confusion_dt)
        skplt.metrics.plot_confusion_matrix(ytest, y_pred_dt ,text_fontsize="large")
        plt.savefig('Confusion Matrix DTnew.png')
        
        class_report_dt = classification_report(ytest,y_pred_dt)
        print("Classification report - Decision Tree:")
        print(class_report_dt)
        
        acc_dt = accuracy_score(ytest, y_pred_dt)
        #print("Accuracy - Decision Tree:")
        #print(acc_dt)
        
        # Precision / Recall / Threshold     
        #y_scores_dt = dt.predict_proba(xtestvec)[:,1] # das geht so nicht
        #print(y_scores_dt)
        # geht nicht, weil keine probabilities
        #p_dt, r_dt, thresholds_dt = precision_recall_curve(ytest, y_scores_dt)
        #plot_precision_recall_vs_threshold(p_dt,r_dt,thresholds_dt)
        #plt.savefig('prthreshold_DT.png')



# -----------------------------------------------------------------------------



# Stratified 10-Fold Cross Validation
# consider the settings of ML algorithms and TF-IDF

strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True,random_state=0)


def classification_report_with_accuracy_score(y_true, y_pred):
    print(classification_report(y_true, y_pred))# print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score


# all Pipelines for Cross Val TFIDF

pipeline_nb = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize,min_df=2,ngram_range=(1, 4),stop_words=my_stops_list)),
        ('clf', MultinomialNB())
        ])

pipeline_svm = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize,min_df=2,ngram_range=(1, 4),stop_words=my_stops_list)),
        ('clf', SVC(kernel="linear",random_state=0,probability=True))
        ])

pipeline_rf = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize,min_df=2,ngram_range=(1, 4),stop_words=my_stops_list)),
        ('clf', RandomForestClassifier(bootstrap=False, random_state=0))
        ])

pipeline_logreg = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize_nostemm,min_df=2,stop_words=my_stops_list)),
        ('clf', LogisticRegression(random_state=0))
        ])

pipeline_dt = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize_nostemm,min_df=2,ngram_range=(1, 4),stop_words=my_stops_list)),
        ('clf', DecisionTreeClassifier(random_state=0))
        ])


# -----------------------------------------------------------------------------

# Cross Val D2V


xvec = vectorizer.fit_transform(x)
y = y.astype('int')
#print(y)

x_D2V_tagged = tag_docs(x)
x_label, xvec_D2V = vec_for_learning(model, x_D2V_tagged)
#print(xvec_D2V)


#scores_svm_D2V = cross_validate(svm, xvec_D2V, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - SVM - D2V:")
#print(scores_svm_D2V)

#scores_rf_D2V = cross_validate(rf, xvec_D2V, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - RF - D2V:")
#print(scores_rf_D2V)

#scores_logreg_D2V = cross_validate(logreg, xvec_D2V, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - LogReg - D2V:")
#print(scores_logreg_D2V)

#scores_dt_D2V = cross_validate(dt, xvec_D2V, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - DT - D2V:")
#print(scores_dt_D2V)


# -----------------------------------------------------------------------------

# Cross Val TFIDF


# Cross Val Results for Naive Bayes

#scores_nb = cross_validate(pipeline_nb, x, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - Naive Bayes:")
#print(scores_nb)


# Cross Val Results for Support Vector Machine

#scores_svm = cross_validate(pipeline_svm, x, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - Support Vector Machine:")
#print(scores_svm)


# Cross Val Results for Random Forest
 
#scores_rf = cross_validate(pipeline_rf, x, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - Random Forest:")
#print(scores_rf)  

  
# Cross Val Results for Logistic Regression

#scores_logreg = cross_validate(pipeline_logreg, x, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - Logistic Regression:")
#print(scores_logreg)

    
# Cross Val Results for Decision Tree

#scores_dt = cross_validate(pipeline_dt, x, y, cv=strat_k_fold, scoring=make_scorer(classification_report_with_accuracy_score), return_train_score=False)
#print("Cross Validation Result - Decision Tree:")
#print(scores_dt)



# -----------------------------------------------------------------------------

# Learning Curve
#strat_x_fold = StratifiedKFold(n_splits=10, shuffle=True,random_state=800)



#lcurve_res = learning_curve(nb, xvec, y, train_sizes=[0.1, 0.33, 0.55, 0.78, 1. ], cv=strat_k_fold, scoring=make_scorer(accuracy_score))

# Working(15.02.2020)
#lcurve_fold_res = learning_curve(nb, xvec, y, train_sizes=np.linspace(0.00001, 1, num=10), cv=strat_2_fold, scoring="recall")

#print (lcurve_fold_res)

#print("Mean:")
#lcurve_mean_res = np.average(lcurve_fold_res[2], axis=1)
#print(lcurve_mean_res)

learn_curve = False

if learn_curve:
    
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(logreg, xvec_D2V, y, train_sizes=np.linspace(0.00001, 1, num=10), cv=strat_k_fold, scoring="precision")
    
    
    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    
    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross validation score")
    
    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
    
    # Create plot
    plt.title("Learning curve")
    plt.xlabel("Training set size"), plt.ylabel("Precision score"), plt.legend(loc="best")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, 1100, step=100))
    plt.tight_layout()
    plt.show()



# -----------------------------------------------------------------------------

#Kappa calculation
#dataforkappa = pd.read_csv("testdatalubeoil_Berg.csv", sep=";", engine="python")
#kappascore = cohen_kappa_score(dataforkappa.myClassification,dataforkappa.expertClassification)
#print("Cohen's Kappa:")
#print(kappascore)







    