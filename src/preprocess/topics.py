import fire
import sys
import tqdm
import pandas as pd
import math
import numpy as np
import kgbench as kg
from collections import Counter
import matplotlib.pyplot as pl
import random as rd
from operator import itemgetter
import torch
import torch.nn.functional as F
from utils import RDF_NUMBER_TYPES, get_relevant_relations, add_triple, get_p_types, ALL_LITERALS, POTENTIAL_TEXT_TYPES
from kgbench.load import Data
from typing import List, Sequence, Tuple

from kgbench import load, tic, toc, d
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils import URI_PREFIX
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import datetime
# from utils import URI_PREFIX
# from preprocess.binning import delete_empty_bin_types
from nltk.corpus import stopwords
import gensim
from gensim.utils import simple_preprocess
import nltk
import torch
#nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim.corpora as corpora
import pyLDAvis.gensim_models as gensimvis
import pickle 
import pyLDAvis
import os
import re
import numpy as np

from os.path import join
from pathlib import Path

import pandas as pd
import gzip, base64, io, sys, warnings, wget, os, random

import torch
from utils import get_relevant_relations, IMAGE_TYPES
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def delete_empty_bin_types(data: Data, num_bins:int)-> Data:
    to_delete= []
    for d in data.i2e[-num_bins:]:
        filtered = data.triples[data.triples[:,2]==data.e2i[d]]
        if len(filtered) == 0:
            to_delete.append(data.e2i[d])

    if(len(to_delete)!=0):
        #to_delete_i = [data.e2i[d] in]
        print(f"deleting relations {to_delete}, since no occurences are given")
                #create new e mapping
        new_e2i = {}
        new_i2e = []

        for i in range(len(data.i2e)):
            if i not in to_delete:
                nt = torch.tensor(len(new_i2e), dtype=torch.int32)
                it = torch.tensor(i, dtype=torch.int32)
                new_e2i[data.i2e[it]] = nt
                new_i2e.append(data.i2e[it])
                # apply new mapping for triples
        for t in data.triples:
            t[0] = new_e2i[data.i2e[t[0]]]
            #t[1] = torch.tensor(data.r2i[data.i2r[t[1].numpy()]], dtype=torch.int32)
            t[2] = new_e2i[data.i2e[t[2]]]

            # create new train & withheld

            #update metedata
        data.num_entities = len(new_i2e)

        data.i2e = new_i2e
        data.e2i = new_e2i
        print('done deleteing')
        print(data.name)
    return data    

def get_stopword_list(languages = ['dutch','spanish','french','portuguese','english']):
    stopword_list= []
    for language in languages:
        stopword_list.extend(stopwords.words(language))
    return stopword_list

def remove_stopwords(word_array, stopword_list):
    return [word for word in word_array
             if word not in stopword_list]

def LDA_topic_assignment(data, num_topics=10, min_mean_word_count = 3, max_assigned_topic = 3, min_topic_relevance = 0.1 ):   
    relevent_relations = get_relevant_relations(
        data, relevant_types=POTENTIAL_TEXT_TYPES)
    #print(num_bins)
    for b in range(num_topics):
        o = (f'{URI_PREFIX}entity#topic{b}', f'{URI_PREFIX}datatype#topics')
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1

    stopword_list = get_stopword_list()
    for r in relevent_relations:
        df = pd.DataFrame(data.triples[data.triples[:,1]== r], columns=['s','p','o'])
        df['text'] = df['o'].apply(lambda t: data.i2e[t][0])
        df['type'] = df['o'].apply(lambda t: data.i2e[t][1])
    
        # delete "none type" polygons 
        df['text'] = df['text'].apply(lambda t:'' if re.match('(MULTIPOLYGON|POLYGON)',t) else t)
        mean_num_words = df['text'].str.count(r'([\w\:\.\/]{3,})').mean()
        #print(mean_num_words)
        if mean_num_words > min_mean_word_count:
            p = f'{URI_PREFIX}predicat#topics{r}'
            new_id = len(data.i2r)
            data.r2i[p] = new_id
            data.i2r.append(p)
            data.num_relations += 1



            df['text_preprocessed'] = df['text'].apply(lambda t: gensim.utils.simple_preprocess(t, deacc=True))
            df['text_preprocessed'] = df['text_preprocessed'].apply(lambda t: remove_stopwords(t, stopword_list=stopword_list))

            data_words = df.text_preprocessed.values.tolist()
            #data_words = list(sent_to_words(text_data))
            print(data_words[:1][0][:30])
            # Create Dictionary
            id2word = corpora.Dictionary(data_words)
            # Term Document Frequency
            corpus = [id2word.doc2bow(text) for text in data_words]
            print(corpus[:1][0][:30])
            num_topics = 10
            lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                                   iterations=100,
                                       id2word=id2word,
                                       num_topics=num_topics,
                                       random_state=42)
            df['vector'] = df['text_preprocessed'].apply(lambda t: id2word.doc2bow(t))
            df['topics'] = df['vector'].apply(lambda t: [x[0] for x in sorted(lda_model.get_document_topics(t), key=lambda x: x[1], reverse=True)[:3] if x[1]>0.1])
            for i in range(num_topics):
                df[f'topic_{i}'] = df['topics'].apply(lambda t: True if i in t else False)
        
            for i in range(num_topics):

                sub = df[df[f'topic_{i}']==True]
                if  len(sub)> 0:
                    sub_df = torch.zeros(len(sub),3,dtype=torch.int32)
                    sub_df[:,0] = torch.tensor(sub.s.tolist(), dtype=torch.int32)
                    #torch.full((len(sub),1),data.r2i[f'{URI_PREFIX}predicat#topics{9}'], dtype=torch.int32)
                    sub_df[:,1] = data.r2i[f'{URI_PREFIX}predicat#topics{r}']
                    sub_df[:,2] = data.e2i[f'{URI_PREFIX}entity#topic{i}', f'{URI_PREFIX}datatype#topics']
                    data.triples = torch.cat((data.triples, sub_df), 0)
    #data = delete_empty_bin_types(data,num_topics)
    return data





def is_image(b64):
    try:
        imgdata = base64.urlsafe_b64decode(b64)
    except:
        # print(f'Could not decode b64 string {b64}')
        return False
    return True

def get_image(b64):
    try:
        imgdata = base64.urlsafe_b64decode(b64)
    except:
        print(f'Could not decode b64 string {b64}')
        return Image.new('RGB', (1, 1))


    try:
        return Image.open(io.BytesIO(base64.urlsafe_b64decode(b64)))
    except:
        return Image.new('RGB', (1, 1))

def eval_image(image, preprocess, model, device= "cpu"):
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)
    return vgg(input_batch).squeeze(0).softmax(0).argmax().item()
    

def VGG_image_classification(data, **kwargs ):
    relevant_relations = get_relevant_relations(data, IMAGE_TYPES)
    df = pd.DataFrame(data.triples[torch.isin(data.triples[:,1],torch.tensor(relevant_relations))], columns = ["s","p","o"])
    vgg = models.vgg16(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg = vgg.to(device)
    vgg.eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Resize(256),  # Resize the image to 256x256 pixels
        # transforms.CenterCrop(224),  # Crop the center 224x224 pixels
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])
    
    df["b64"] = df['o'].apply(lambda x: data.i2e[x][0])
    df["is_image"] = df["b64"].apply(lambda x: is_image(x))
    df = df[df['is_image']==True]
    df["image"] = df['b64'].apply(lambda x: get_image(x))
    df['class']= df["image"].apply(lambda x: vgg(preprocess(x).unsqueeze(0).to(device)).squeeze(0).softmax(0).argmax().item())
    print(df['class'])

    for pred in df['p'].unique(): # use this over relation since loading image can subset df
        sub_df = df[df['p']==pred]
        p = f'{URI_PREFIX}predicat#img-class-{pred}'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1
        for c in sub_df["class"].unique():
            o = (f'{URI_PREFIX}entity#img-class-{c}-{pred}',
            f'{URI_PREFIX}datatype#img-class')
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1

            
        sub_df['new_o'] = sub_df["class"].apply(lambda c: data.e2i[(f'{URI_PREFIX}entity#img-class-{c}-{pred}',
            f'{URI_PREFIX}datatype#img-class')])
        sub_df['new_p'] = sub_df["class"].apply(lambda f: data.r2i[f'{URI_PREFIX}predicat#img-class-{pred}'])
        ten = torch.tensor(sub_df[['s','new_p','new_o']].values.astype(np.int32), dtype= torch.int32)
        data.triples = torch.cat((data.triples, ten), 0)
    return data
