import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.samplers import WideSampler
from pyrdf2vec.walkers import HALKWalker

import numpy as np
import kgbench

import fire
import sys
import tqdm
import pandas as pd
import numpy as np
import kgbench as kg
from collections import Counter
import matplotlib.pyplot as pl
import random as rd
from operator import itemgetter
import torch
import torch.nn.functional as F
#from experiments.rgcn import RGCN, adj, sum_sparse, enrich
from kgbench import load, tic, toc, d

from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.samplers import PageRankSampler

import asyncio
import pickle
import time
from typing import List, Sequence, Tuple

import attr
import pyrdf2vec as r2v
from pyrdf2vec.embedders import Embedder, Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.typings import Embeddings, Entities, Literals, SWalk
from .bfs_random_walker import BFSRandomWalker
from pyrdf2vec.walkers import RandomWalker, Walker
from utils.data_utils import data_to_kg, extract_ents


class RDF2Vec():
    
    def __init__(self, data, embedding_name="Word2Vec", embedding_args={"workers": 4, "epochs": 40},
                 walker_name="BFSRandomWalker", walker_args={"max_depth": 2, "with_reverse":True, "random_state":42}):
        self.data = data
        torch.cuda.empty_cache()
        embedder = getattr(
            r2v.embedders, embedding_name)(**embedding_args)
        if walker_name=="BFSRandomWalker":
            walker =BFSRandomWalker(**walker_args)
        else:
            walker = getattr(
                r2v.walkers, walker_name)(**walker_args)
        self.transformer: RDF2VecTransformer = RDF2VecTransformer(
            embedder, walkers=[walker], verbose=1)

    def fit_transform(self) -> Tuple[Embeddings, Embeddings, Embeddings]:
        kg = data_to_kg(self.data)
        train_entities, test_entities, train_target, test_taget = extract_ents(
        self.data)  # extract necessary fields from data
        entities = train_entities + test_entities

        embeddings = self.transformer.fit_transform(kg, entities)[0]
        train_embeddings = embeddings[: len(train_entities)]
        test_embeddings = embeddings[len(train_entities):]
        return embeddings, train_embeddings, test_embeddings



class RDF2Vec2():

    def __init__(self, data, embedding_name="Word2Vec", embedding_args={"workers": 4, "epochs": 40},
                 walker_name="RandomWalker", walker_args={"max_depth": 2}):
        self.data = data
        embedder = getattr(
            r2v.embedders, embedding_name)(**embedding_args)
        walker = getattr(
            r2v.walkers, walker_name)(**walker_args)
        self.transformer: RDF2VecTransformer = RDF2VecTransformer(
            embedder, walkers=[walker], verbose=0)

    def fit_transform(self) -> Tuple[Embeddings, Embeddings, Embeddings]:
        kg = data_to_kg(self.data)
        train_entities, test_entities, train_target, test_taget = extract_ents(
        self.data)  # extract necessary fields from data
        entities = train_entities + test_entities

        embeddings = self.transformer.fit_transform(kg, entities)[0]
        train_embeddings = embeddings[: len(train_entities)]
        test_embeddings = embeddings[len(train_entities):]
        return embeddings, train_embeddings, test_embeddings



