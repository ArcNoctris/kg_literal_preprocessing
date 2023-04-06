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
from utils import RDF_NUMBER_TYPES, get_relevant_relations, add_triple, get_p_types, ALL_LITERALS, ALL_BUT_NUMBER
from kgbench.load import Data
from typing import List, Sequence, Tuple

from kgbench import load, tic, toc, d
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils import URI_PREFIX,POTENTIAL_TEXT_TYPES, RDF_DATE_TYPES, IMAGE_TYPES, GEO_TYPES, NONE_TYPES
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils import delete_r


def delete_number_literals(data:Data,**kwargs)-> Data:
    rr = get_relevant_relations(data,RDF_NUMBER_TYPES)
    return delete_r(data, torch.tensor(rr))

def delete_all_literals(data:Data, **kwargs)-> Data:
    rr = get_relevant_relations(data,ALL_LITERALS)
    return delete_r(data, torch.tensor(rr))

def delete_all_but_numbers(data:Data, **kwargs) -> Data:
    rr = get_relevant_relations(data,ALL_BUT_NUMBER)
    return delete_r(data, torch.tensor(rr))

def delete_date_literals(data:Data, **kwargs) -> Data:
    rr = get_relevant_relations(data,RDF_DATE_TYPES)
    return delete_r(data, torch.tensor(rr))

def delete_image_literals(data:Data, **kwargs) -> Data:
    rr = get_relevant_relations(data,IMAGE_TYPES)
    return delete_r(data, torch.tensor(rr))

def delete_geo_literals(data:Data, **kwargs) -> Data:
    rr = get_relevant_relations(data,GEO_TYPES)
    return delete_r(data, torch.tensor(rr))

def delete_none_literals(data:Data, **kwargs) -> Data:
    rr = get_relevant_relations(data,POTENTIAL_TEXT_TYPES)
    return delete_r(data, torch.tensor(rr))

def delete_text_literals(data:Data, **kwargs) -> Data:
    rr = get_relevant_relations(data,POTENTIAL_TEXT_TYPES)
    return delete_r(data, torch.tensor(rr))
