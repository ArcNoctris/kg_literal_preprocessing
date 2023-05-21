import kgbench as kg
from utils import Data
import numpy as np
import re
from typing import Union
from utils import get_relevant_relations, RDF_DECIMAL_TYPES, ensure_data_symmetry
BASE_FILE_PATH = "data/raw"
import os
import pickle

def amplus(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    if pickle_exist('amplus', final=final, torch=torch, prune_dist=prune_dist):
        data = load_pickle('amplus', final=final, torch=torch, prune_dist=prune_dist)
    else:
        pickle_name = f'amplus_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load('amplus', final=final, torch=torch, prune_dist=prune_dist)
        print("clean up data (Transform decimals)")
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
                    data.e2i[data.i2e[triple[2]]] = int(triple[2])
            data = ensure_data_symmetry(data)
        print("save file...")
        with open(f'{BASE_FILE_PATH}/{pickle_name}.pickle', "wb") as f:
            pickle.dump(data, f)      
    return data

def dmgfull(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    if pickle_exist('dmgfull', final=final, torch=torch, prune_dist=prune_dist):
        data = load_pickle('dmgfull', final=final, torch=torch, prune_dist=prune_dist)
    else:
        pickle_name = f'dmgfull_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load('dmgfull', final=final, torch=torch, prune_dist=prune_dist)
        print("save file...")
        with open(f'{BASE_FILE_PATH}/{pickle_name}.pickle', "wb") as f:
            pickle.dump(data, f)
    return data


def dmg777k(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    if pickle_exist('dmg777k', final=final, torch=torch, prune_dist=prune_dist):
        data = load_pickle('dmg777k', final=final, torch=torch, prune_dist=prune_dist)
    else:
        pickle_name = f'dmg777k_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load('dmg777k', final=final, torch=torch, prune_dist=prune_dist)
        print("save file...")
        with open(f'{BASE_FILE_PATH}/{pickle_name}.pickle', "wb") as f:
            pickle.dump(data, f)
    return data


def mdgenre(final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    if pickle_exist('mdgenre', final=final, torch=torch, prune_dist=prune_dist):
        data = load_pickle('mdgenre', final=final, torch=torch, prune_dist=prune_dist)
    else:
        pickle_name = f'mdgenre_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
        data = _load('mdgenre', final=final, torch=torch, prune_dist=prune_dist)
        print("save file...")
        with open(f'{BASE_FILE_PATH}/{pickle_name}.pickle', "wb") as f:
            pickle.dump(data, f)
    return data


def _load(dataset_name="dmg777k", final=True, torch=True, prune_dist=None, **kwargs) -> Data:
    # if dataset_name in ['amplus', 'dblp', 'dmgfull', 'dmg777k', 'mdgenre']:
    # ignore warning, datatype loaded here is extended by datatypes defined in the framework specific Data type 
    data:Data = kg.load(dataset_name, final, torch, prune_dist=prune_dist) # type: ignore
    #fixing https://github.com/pbloem/kgbench-loader/issues/2
    if prune_dist == None:
        clean_e2i = {}
        for e in data.e2i.keys():
            clean_e2i[e[1]] = e[0]
        data.e2i = clean_e2i
    print("ensure data symmetry...")
    data = ensure_data_symmetry(data)
    return data

def pickle_exist(dataset_name="dmg777k", final=True, torch=True, prune_dist=None)-> bool:
    pickle_name = f'{dataset_name}_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
    return os.path.exists(f'{BASE_FILE_PATH}/{pickle_name}.pickle')

def load_pickle(dataset_name="dmg777k", final=True, torch=True, prune_dist=None)-> Data:
    pickle_name = f'{dataset_name}_{"final" if final else ""}_{"torch" if torch else ""}_{prune_dist}'
    with open(f'{BASE_FILE_PATH}/{pickle_name}.pickle', "rb") as f:
        data:Data = pickle.load(f)
    return data



