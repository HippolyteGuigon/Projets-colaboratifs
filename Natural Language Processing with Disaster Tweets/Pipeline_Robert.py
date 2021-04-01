import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import json
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
import re
import torch
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
import torch
import torch.nn as nn
from transformers import BertModel
warnings.filterwarnings("ignore")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('words', quiet=True) 
from transformers import AdamW, get_linear_schedule_with_warmup
import random
import time
loss_fn = nn.CrossEntropyLoss()
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["location"].fillna(" ", inplace=True)
train["keyword"].fillna(" ", inplace=True)

from transformers import (
    BertForSequenceClassification,
#     TFBertForSequenceClassification, 
                          BertTokenizer,
#                           TFRobertaForSequenceClassification,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                         AdamW)

# BERT
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
                                                                num_labels = 2, # The number of output labels--2 for binary classification.
                                                                                # You can increase this for multi-class tasks.   
                                                                output_attentions = False, # Whether the model returns attentions weights.
                                                                output_hidden_states = False # Whether the model returns all hidden-states.
                                                          )
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                                                           

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Create a function to tokenize a set of texts
MAX_LEN = 6851


class BertClassifier(nn.Module):
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert=False):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        # Instantiate BERT model
        self.bert = BertTokenizer.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(H, D_out)
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


def text_preprocessing(s):
    """
    - Lowercase the sentence
    - Change "'t" to "not"
    - Remove "@name"
    - Isolate and remove punctuations except "?"
    - Remove other special characters
    - Remove stop words except "not" and "can"
    - Remove trailing whitespace
    """
    s = s.lower()
    # Change 't to 'not'
    s = re.sub(r"\'t", " not", s)
    # Remove @name
    s = re.sub(r'(@.*?)[\s]', ' ', s)
    # Isolate and remove punctuations except '?'
    s = re.sub(r'([\'\"\.\(\)\!\?\\\/\,])', r' \1 ', s)
    s = re.sub(r'[^\w\s\?]', ' ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove stopwords except 'not' and 'can'
    s = " ".join([word for word in s.split()
                  if word not in stopwords.words('english')
                  or word in ['not', 'can']])
    # Remove trailing whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s



def get_auc_CV(model):
    """
    Return the average AUC score from cross-validation.
    """
    # Set KFold to shuffle data before the split
    kf = StratifiedKFold(5, shuffle=True, random_state=1)

    # Get AUC scores
    auc = cross_val_score(
        model, X_train_tfidf, y_train, scoring="roc_auc", cv=kf)

    return auc.mean()


class BertModel:
    
    def __init__(self, proportion_test):
        
        self.pt = proportion_test
        
    def clean(self):
        
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        train["text"] = train["text"].apply(text_preprocessing)
        test["text"] = test["text"].apply(text_preprocessing)
        
        return pd.DataFrame(train), pd.DataFrame(test)
        
        
    def split_and_tfidf(self):
        
        
        
        X = self.clean()[0]["text"] + self.clean()[0]["location"].fillna(" ") + self.clean()[0]["keyword"].fillna(" ")
        y = self.clean()[0]["target"]
        
        X_train, X_val, y_train, y_test = train_test_split(X, y, test_size=self.pt)
        
        
        # Preprocess text
        X_train_preprocessed = np.array([text_preprocessing(text) for text in X_train])
        X_val_preprocessed = np.array([text_preprocessing(text) for text in X_val])

        # Calculate TF-IDF
        tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                         binary=True,
                         smooth_idf=False)
        X_train_tfidf = tf_idf.fit_transform(X_train)
        X_val_tfidf = tf_idf.transform(X_val)
        
        print(np.array(y_train).shape, np.array(X_train).shape)
        
        return X_train, X_val, y_train, y_test, X_train_tfidf, X_val_tfidf
    
    def concatenate(self, data):
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        
        all_tweets = np.concatenate([self.clean()[0].text.values + self.clean()[0].location.fillna(" ").values + self.clean()[0].keyword.fillna(" ").values, self.clean()[1].text.fillna(" ").values + self.clean()[1].keyword.fillna(" ").values + self.clean()[1].location.fillna(" ").values])

        # Encode our concatenated data
        encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in all_tweets]
        
        max_len = max([len(sent) for sent in encoded_tweets])
        print('Max length: ', max_len)
        
        return encoded_tweets
    
    def preprocessing_for_bert(self, data):

        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = tokenizer.encode_plus(
                text=text_preprocessing(sent),  # Preprocess sentence
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                max_length=64,                  # Max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length
                #return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True      # Return attention mask
                )
        
            # Add the outputs to the lists
            input_ids.append(encoded_sent.get('input_ids'))
            attention_masks.append(encoded_sent.get('attention_mask'))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks
    
    def get_train_val(self):
        
        MAX_LEN = 64
        train_inputs, train_masks = self.preprocessing_for_bert(self.split_and_tfidf()[0])
        
        val_inputs, val_masks = self.preprocessing_for_bert(self.split_and_tfidf()[1])
        # Convert other data types to torch.Tensor
        train_labels = torch.tensor(np.array(self.split_and_tfidf()[2]))
        val_labels = torch.tensor(np.array(self.split_and_tfidf()[3]))
        
        # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
        batch_size = 32
        
        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
        
        return train_data, train_sampler, train_dataloader, val_data, val_sampler, val_dataloader