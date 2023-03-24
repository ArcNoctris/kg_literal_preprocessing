import kgbench as kg
from kgbench.load import Data
import numpy as np
import re
from utils import get_relevant_relations, RDF_DECIMAL_TYPES

def amplus(final=True,  torch = True, prune_dist = 5,**kwargs):
    data = _load('amplus',final = final, torch = torch,prune_dist = prune_dist)
    # clean up decimals (bad data quality)
    rr = get_relevant_relations(data, RDF_DECIMAL_TYPES)
    for r in rr:
        df_sub = data.triples[data.triples[:,1]==r]
        for d in df_sub:
            m = re.search(r'[0-9]+\.?[0-9]*',data.i2e[d[2]][0])
            new_string = '0'
            if m != None:
                new_string = m.group()
            data.e2i.pop(data.i2e[d[2]])
            data.i2e[d[2]] = (new_string,data.i2e[d[2]][1])
            data.e2i[data.i2e[d[2]]] =d[2]
    return data
def dmgfull(final=True,  torch = True, prune_dist = 5,**kwargs):
    return _load('dmgfull',final = final, torch = torch,prune_dist = prune_dist)
def dmg777k(final=True,  torch = True, prune_dist = 5,**kwargs):
    return _load('dmg777k',final = final, torch = torch,prune_dist = prune_dist)
def mdgenre(final=True,  torch = True, prune_dist = 5,**kwargs):
    return _load('mdgenre',final = final, torch = torch,prune_dist = prune_dist)
def _load(dataset_name="dmg777k", final=True,  torch = True, prune_dist = 5,**kwargs) -> Data:
    if dataset_name in ['amplus','dblp','dmgfull','dmg777k','mdgenre']:
        return kg.load(dataset_name,final,torch,prune_dist=prune_dist)