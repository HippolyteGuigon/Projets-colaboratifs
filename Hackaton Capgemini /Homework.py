import json
import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings(action='ignore')
from string import punctuation
from nltk import word_tokenize
from nltk import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
import tkinter as tk
from tkinter import ttk
import warnings
warnings.filterwarnings("ignore")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True)


train = pd.read_csv("train.csv")

title = train["title"]
description = train["description"]

def cleaning_texte(texte):
    
        #Switiching to lower character the corpus
        corpus1 = [review.lower() for review in texte if type(review)==str]  

        #Remove special character
        characters_to_remove = ["@", "/", "#", ".", ",", "!", "?", ";", "(", ")", "-", "_","’","'", "\"", ":", "&", "\n"]
        transformation_dict = {initial:" " for initial in characters_to_remove}
        corpus2 = [review.translate(str.maketrans(transformation_dict)) for review in corpus1]

        #Tokenized the text
        tokenized_corpus = [nltk.word_tokenize(review) for review in corpus2]

        #Delete foreign world
        words = set(nltk.corpus.words.words())
        english_corpus = [[w for w in tokenized_corpus[i] if w in words or not w.isalpha()]
                    for i in range(len(tokenized_corpus))]

        #Stoping world
        stop_words = nltk.corpus.stopwords.words("english")
        new_corpus = [[token for token in english_corpus[i] if token not in stop_words] 
                  for i in range(len(english_corpus))]

        #Lematizing
        lmtzr = WordNetLemmatizer()
        lematized = [[lmtzr.lemmatize(word) for word in new_corpus[i]] 
              for i in range(len(new_corpus))]
        
        print(lematized)

        #Remove number
        #if number == True :
        #    number_corpus = [[w if not any(j.isdigit() for j in w)  else 'Token_number' for w in lematized[i]]
        #             for i in range(len(lematized)) ]
        #alternatice : w.isdigit() instead of any(j.isdigit() for j in w)
        #    return number_corpus

        #else :
        #    return lematized
    


class Affichage:
    
    def __init__(self, restaurant_name):
        
        self.r = restaurant_name
        
    def texte(self):
        
        #The Reviews and information are being imported from the json file
       
        with open('sauvegardeFinaleGrade4.json') as json_data:
            data_dict = json.load(json_data)
        
        df = pd.DataFrame(data_dict)
        
        #Taking the information about the very restaurant selected 
    
        if str(self.r) not in df["Restaurant"].unique():
            print("The restautant you entered is not in the list")
        else:
            df = df[df["Restaurant"] == str(self.r)]
        
        return df["Review"]
        
    def cleaning_lemmatize(self, number = True):
            
        #Switiching to lower character the corpus
        corpus1 = [review.lower() for review in self.texte()]  

        #Remove special character
        characters_to_remove = ["@", "/", "#", ".", ",", "!", "?", ";", "(", ")", "-", "_","’","'", "\"", ":", "&", "\n"]
        transformation_dict = {initial:" " for initial in characters_to_remove}
        corpus2 = [review.translate(str.maketrans(transformation_dict)) for review in corpus1]

        #Tokenized the text
        tokenized_corpus = [nltk.word_tokenize(review) for review in corpus2]

        #Delete foreign world
        words = set(nltk.corpus.words.words())
        english_corpus = [[w for w in tokenized_corpus[i] if w in words or not w.isalpha()]
                    for i in range(len(tokenized_corpus))]

        #Stoping world
        stop_words = nltk.corpus.stopwords.words("english")
        new_corpus = [[token for token in english_corpus[i] if token not in stop_words] 
                  for i in range(len(english_corpus))]

        #Lematizing
        lmtzr = WordNetLemmatizer()
        lematized = [[lmtzr.lemmatize(word) for word in new_corpus[i]] 
              for i in range(len(new_corpus))]

        #Remove number
        if number == True :
            number_corpus = [[w if not any(j.isdigit() for j in w)  else 'Token_number' for w in lematized[i]]
                     for i in range(len(lematized)) ]
        #alternatice : w.isdigit() instead of any(j.isdigit() for j in w)
            return number_corpus

        else :
            return lematized
    
    def mapping(self):
        
        #This function makes the count of every word in a certain number of reviews
        
        vocab = list(set([item for sentence in self.cleaning_lemmatize() for item in sentence]))
        vocab_freq = {word:0 for word in vocab}
      
        #Counting the occurence of each word
        
        for sentence in self.cleaning_lemmatize():
            for word in sentence:
                if word in vocab: 
                    vocab_freq[word] +=1
        vocab_freq = {k: v for k, v in sorted(vocab_freq.items(), key=lambda item: item[1], reverse=True)}
    
        wordSet = set().union(*self.cleaning_lemmatize())
        list(wordSet)
        
        #Puting those occurences in a dictionnary
        
        wordDict = []

        for i in range(len(self.cleaning_lemmatize())):
            wordDict.append(dict.fromkeys(wordSet, 0))
    
        for i, sent in enumerate(self.cleaning_lemmatize()): 
            for word in sent:
                wordDict[i][word]+=1
                
        #Creates an index to better represent the occurences
    
        index = []

        for i in range(len(self.cleaning_lemmatize())):
            index.append("Review" + " " + str(i))
            
        #Defining a dataframe to store the results 
        
        df_bow = pd.DataFrame(wordDict, index=index)
        
        return df_bow
    
    def get_tfidf_Matrix(self):
        
        #Creating a matrix to see the frequency of each word
        
        corpus_clean = [' '.join(sentence) for sentence in self.cleaning_lemmatize()]
        vectorizer = TfidfVectorizer(stop_words='english')
        vect_corpus = vectorizer.fit_transform(corpus_clean)
        feature_names = np.array(vectorizer.get_feature_names())

        df_tfidf = pd.DataFrame(vect_corpus.todense(), columns = feature_names)
        return df_tfidf
    
    def graph(self):
        
        #Creating graphs to see the results 
        
        df_mean = (self.mapping()).mean().sort_values(ascending=False).to_frame(name='occurence mean')
        
        dict_words_bow = df_mean[df_mean['occurence mean'] != 0].to_dict()['occurence mean']
        
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        
        #Graph to see the frequency of the 20 most written words 
        
        g0 = df_mean[:20].plot(ax=ax[0,0], kind="bar")
        g0.set_title("Occurence of words for the first 20 reviews", fontsize=15)
        g0.set_ylabel("Occurence", fontsize=15)
    
        #Heatmap to visualize the words that appear most 
        
        g1 = sns.heatmap(self.mapping().iloc[:30, :30], cmap='YlGn', ax=ax[0, 1])
        g1.set_ylabel("Review number", fontsize=15)
        g1.set_title("Document Visualisation with BOW Representation for the first 30 reviews", fontsize=15)

        #Graph to see the length of reviews for the restaurant
        
        g2 = sns.distplot(df['Review'].apply(len), ax=ax[1, 0])
        g2.set_title("Length review distribution", fontsize=15)
        plt.xticks(range(0,2000,250))
    
        #Barplot to see the distribution of grading for the restaurant
        
        X_ratings = df["Grade"].value_counts()
        g3 = sns.barplot(X_ratings.index,X_ratings,alpha=0.8, ax=ax[1, 1])
        g3.set_title("Number of stars for restaurant review", fontsize=15)
        g3.set_xlabel("Number of reviews for a given score", fontsize=15)
        g3.set_ylabel("Grade", fontsize=15)
        
        #Displaying the Wordcloud graph
        
        wordcloud = WordCloud(height=600, width=800, background_color="white", colormap='inferno', max_words=100)
        wordcloud.generate_from_frequencies(frequencies=dict_words_bow)
        plt.figure(figsize=(20,15))
        plt.title("Wordcloud", fontsize=15)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
        