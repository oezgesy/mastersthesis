# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:02:41 2020

@author: z003rukn
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

data = pd.read_csv("word2vec_vocabulary.csv", sep=";", engine="python")
#print("DATA is:")
#print(data)

mydata = data['vocabulary'].astype(str).values.tolist()

punct = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~•’”“–°"
my_stops = pd.read_csv("Stopwords.csv", sep=";", engine="python")
my_stops_list = my_stops.Stopwords.values.tolist()

#def tokenize(text):
#    text = "".join([ch for ch in text if ch not in punct])
#    tokens = word_tokenize(text)
#    return tokens

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


#tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(mydata)]
tagged_data = [TaggedDocument(words=tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(mydata)]
#print(tagged_data)



max_epochs = 40
vec_size = 300
alpha = 0.025



model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=2, dm=1,window=5)
build_new_model=False
    
if build_new_model:

    model.build_vocab(tagged_data)
    
    for epoch in range(max_epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        # decrease learning rate
        model.alpha-=0.0002
        # fix learning rate, no decay
        model.min_alpha=model.alpha
    
    model.save("d2v_window5.model")
    print("Model Saved")
else:
    model = Doc2Vec.load("d2v.model")



my_words=["lubricant","lube", "steel", "stainless", "filters", "piping","oil","supply"]
for word in my_words:
    print("--------------------------------------------------------")
    print("Most similar to words:" + str(word))
    similar_words = model.wv.most_similar(positive=[word],topn=5)
    print(similar_words)

similar_words_pd = pd.DataFrame(similar_words)
#writer = pd.ExcelWriter("C:/Users/z003rukn/Desktop/MA_Results/similar_words.xlsx", engine="xlsxwriter")
#similar_words_pd.to_excel(writer, sheet_name='Confusion_NB')


#my_doc = "lube oil piping shall be made of stainless steel".split()
#my_doc = "The pump and piping shall not be included".split()
my_doc = "Lube Oil System All lube oil supply piping shall be 18-8 stainless".split()



my_doc_vector = model.infer_vector(my_doc)
sims = model.docvecs.most_similar([my_doc_vector],topn=5)
print("Most similar to doc: " + str(my_doc))
print(sims)

for sim_doc in sims:
    print(str(mydata[int(sim_doc[0])]))








