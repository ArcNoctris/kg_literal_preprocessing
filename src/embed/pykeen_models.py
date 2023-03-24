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
FILEPATH = "data/preprocessed"
def data_to_pykeen(data):
    if not os.path.exists(f'{FILEPATH}/{data.name}.tsv.gz'):
        print('pykeen file does not exist. Writing pykeen file...')
        df = pd.DataFrame(columns=["h","r","t"])
        
        dfs = []
        for d in data.triples:
            dfs.append(pd.DataFrame(
                [[
                data.i2e[d[0]][0],
                data.i2r[d[1]],
                data.i2e[d[2]][0]
                ]],
                columns=["h","r","t"]
            ))
        df = pd.concat(dfs,ignore_index=True)
        df.to_csv(f'{FILEPATH}/{data.name}.tsv.gz', 
                index=False,
                sep="\t" ,
                compression="gzip")
    return  TriplesFactory.from_path(f'{FILEPATH}/{data.name}.tsv.gz')
        
        


class PykeenEmbedder():
    def __init__(self, embedder, data, optimizer, optimizer_args,
                 train_loop_type, train_loop_args):
        self.data = data
        

        
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
                 train_loop_type="SLCWA", train_loop_args={"num_epochs": 5,'batch_size':256}):
        super().__init__(models.RGCN, data,optimizer,optimizer_args,train_loop_type,train_loop_args)



        
        


