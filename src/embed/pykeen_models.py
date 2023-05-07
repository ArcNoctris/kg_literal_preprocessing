from pykeen import models
from torch.optim import Adam
from pykeen.training import SLCWATrainingLoop
import pandas as pd
from pykeen.triples import TriplesFactory
from typing import List, Sequence, Tuple
from pyrdf2vec.typings import Embeddings
from pykeen.utils import resolve_device
from utils.data_utils import data_to_kg, extract_ents
import os
import torch
FILEPATH = "data/preprocessed"
def data_to_pykeen(data):
    if not os.path.exists(f'{FILEPATH}/{data.name}.tsv.gz'):
        print('pykeen file does not exist. Writing pykeen file...')
        print(data.name)
        df =pd.DataFrame(data.triples, columns=["h","r","t"])
        df['h'] = df['h'].apply(lambda h:data.i2e[h][0])
        df['r'] = df['r'].apply(lambda r:data.i2r[r])
        df['t'] = df['t'].apply(lambda t:data.i2e[t][0])
        df.to_csv(f'{FILEPATH}/{data.name}.tsv.gz', 
                index=False,
                sep="\t" ,
                compression="gzip")
    return  TriplesFactory.from_path(f'{FILEPATH}/{data.name}.tsv.gz')
        
        


class PykeenEmbedder():
    def __init__(self, embedder, data, optimizer, optimizer_args,
                 train_loop_type, train_loop_args):
        self.data = data
        torch.cuda.empty_cache()
        

        
        self.training_triples_factory = data_to_pykeen(data)
        self.model = embedder(triples_factory=self.training_triples_factory)
        self.model = self.model.to(resolve_device('gpu'))
        optimizer = Adam(params=self.model.get_grad_params())
        self.training_loop = SLCWATrainingLoop(
        model=self.model,
        triples_factory=self.training_triples_factory,
        optimizer=optimizer,
        )
        self.train_loop_args = train_loop_args
    def fit_transform(self) -> Tuple[Embeddings, Embeddings, Embeddings]:
        self.training_loop.train(
        triples_factory=self.training_triples_factory,
        **self.train_loop_args
        )
        embeddings = self.model.entity_representations[0]().detach().cpu().numpy()
        reorder = []
        for e in self.data.i2e:
            reorder.append(self.training_triples_factory.entity_to_id[e[0]])
        embeddings = embeddings[reorder]
        train_entities, test_entities, train_target, test_taget = extract_ents(
        self.data)  # extract necessary fields from data
        train_embeddings = embeddings[self.data.training[:,0]]
        test_embeddings = embeddings[self.data.withheld[:,0]]
        return embeddings, train_embeddings, test_embeddings


class TransE(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="SLCWA", train_loop_args={"num_epochs": 5,'batch_size':256}):
        super().__init__(models.TransE, data,optimizer,optimizer_args,train_loop_type,train_loop_args)

class KG2E(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="SLCWA", train_loop_args={"num_epochs": 5,'batch_size':256}):
        super().__init__(models.KG2E, data,optimizer,optimizer_args,train_loop_type,train_loop_args)

class RGCN(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="SLCWA", train_loop_args={"num_epochs": 5,'batch_size':4.096,"sampler":"schlichtkrull", "gradient_clipping_max_norm":1}):
        super().__init__(models.RGCN, data,optimizer,optimizer_args,train_loop_type,train_loop_args)




class ComplEx(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="SLCWA", train_loop_args={"num_epochs": 5,'batch_size':256}):
        super().__init__(models.ComplEx, data,optimizer,optimizer_args,train_loop_type,train_loop_args)

class SimplE(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="SLCWA", train_loop_args={"num_epochs": 5,'batch_size':256}):
        super().__init__(models.SimplE, data,optimizer,optimizer_args,train_loop_type,train_loop_args)
class DistMult(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="SLCWA", train_loop_args={"num_epochs": 5,'batch_size':256}):
        super().__init__(models.DistMult, data,optimizer,optimizer_args,train_loop_type,train_loop_args)
class RGCN2(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="SLCWA", train_loop_args={"sampler": "schlichtkrull"}):
        super().__init__(models.RGCN, data,optimizer,optimizer_args,train_loop_type,train_loop_args)


class ComplEx2(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="LCWA", train_loop_args={"num_epochs": 5,'batch_size':256}):
        super().__init__(models.ComplEx, data,optimizer,optimizer_args,train_loop_type,train_loop_args)
class DistMult2(PykeenEmbedder):
    def __init__(self, data, optimizer="Adam", optimizer_args={},
                 train_loop_type="LCWA", train_loop_args={"num_epochs": 5,'batch_size':256}):
        super().__init__(models.DistMult, data,optimizer,optimizer_args,train_loop_type,train_loop_args)


        
        


