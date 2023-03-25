import kgbench as kg
from kgbench.load import Data
import numpy as np
import re
from typing import Union
from utils import get_relevant_relations, RDF_DECIMAL_TYPES


def amplus(final=True, torch=True, prune_dist=5, **kwargs) -> Union[Data, None]:
    data = _load('amplus', final=final, torch=torch, prune_dist=prune_dist)
    if data != None and data.triples != None and data.i2e != None and data.e2i != None:
        # clean up decimals (bad data quality)
        relevant_relations = get_relevant_relations(data, RDF_DECIMAL_TYPES)
        for relation in relevant_relations:
            for triple in data.triples[data.triples[:, 1] == relation]:
                regex_search = re.search(
                    r'[0-9]+\.?[0-9]*', data.i2e[triple[2]][0])
                new_string = '0'
                if regex_search != None:
                    new_string = regex_search.group()  # get first result
                data.e2i.pop(data.i2e[triple[2]])
                data.i2e[triple[2]] = (new_string, data.i2e[triple[2]][1])
                data.e2i[data.i2e[triple[2]]] = triple[2]
        return data

def dmgfull(final=True, torch=True, prune_dist=5, **kwargs) -> Union[Data, None]:
    return _load('dmgfull', final=final, torch=torch, prune_dist=prune_dist)


def dmg777k(final=True, torch=True, prune_dist=5, **kwargs) -> Union[Data, None]:
    return _load('dmg777k', final=final, torch=torch, prune_dist=prune_dist)


def mdgenre(final=True, torch=True, prune_dist=5, **kwargs) -> Union[Data, None]:
    return _load('mdgenre', final=final, torch=torch, prune_dist=prune_dist)


def _load(dataset_name="dmg777k", final=True, torch=True, prune_dist=5, **kwargs) -> Union[Data, None]:
    if dataset_name in ['amplus', 'dblp', 'dmgfull', 'dmg777k', 'mdgenre']:
        return kg.load(dataset_name, final, torch, prune_dist=prune_dist)
