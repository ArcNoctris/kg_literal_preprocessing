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
from utils import RDF_NUMBER_TYPES, get_relevant_relations, add_triple, get_p_types
from kgbench.load import Data
from typing import List, Sequence, Tuple

from kgbench import load, tic, toc, d
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils import URI_PREFIX
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def bin_numbers_with_lof(data: Data, num_bins=5):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)

    # convert tensor to numpy array
    tensor_np = data.triples.numpy()

    # create an instance of the LOF algorithm
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

    # fit the LOF model to the feature you want to use to identify outliers
    lof.fit(tensor_np[:, [2]])

    # use the negative_outlier_factor_ attribute to get the outlier scores for each sample
    outlier_scores = lof.negative_outlier_factor_

    # Create a new column in the numpy array to store the outlier scores
    tensor_np = np.hstack((tensor_np, outlier_scores.reshape(-1, 1)))
    threshold = np.percentile(outlier_scores, 1)

    # use the outlier scores to filter out the outliers from the numpy array
    tensor_np = tensor_np[outlier_scores > threshold]

    i = 0
    cumsum = 0
    bins = np.arange(num_bins)

    for r in relevent_relations:
        current_bins = 0
        df_sub = data.triples[data.triples[:, 1] == r]
        # bins = np.digitize(sub_list,np.histogram(sub_list,num_bins,density = True)[1])
        sub_list = np.array([data.i2e[x][0]
                            for x in df_sub[:, 2]], dtype=np.float32)
        # mean = torch.round(torch.mean(df_sub[:,2],dtype = torch.float64)).to(torch.long)

        new = torch.tensor(df_sub[:, 0])

        new_df = torch.ones((new.size()[0], 3), dtype=torch.long)

        new_df[:, 0] = torch.tensor(df_sub[:, 0])  # values at place 0 (iri)
        new_df[:, 1] = data.num_relations + i
        binned_list = torch.from_numpy(np.digitize(
            sub_list, np.histogram(sub_list, num_bins)[1]))
        bi = 0
        while len(np.unique(binned_list)) < num_bins - bi:

            # print(f'{len(np.unique(binned_list))} vs {num_bins-bi}')
            bi += 1
            binned_list = torch.from_numpy(np.digitize(
                sub_list, np.histogram(sub_list, num_bins - bi)[1]))

        current_bins = len(np.unique(binned_list))
        new_df[:, 2] = binned_list
        new_df[:, 2] += (data.num_entities + ((i - 1) * num_bins)) - 1

        data.i2r.append(
            f"https://master-thesis.com/relations#sanity-check-{i}")
        data.r2i[f'https://master-thesis.com/relations#sanity-check-{i}'] = data.num_relations + i

        for e in range(current_bins):
            data.i2e.append(
                (f'https://master-thesis.com/entitys#sanity-check-target-{i}-{e}', 'preprocessed'))
            data.e2i[(
                f'https://master-thesis.com/entitys#sanity-check-target-{i}-{e}', 'preprocessed')] = data.num_entities + cumsum + e

        # i += 1
        # cumsum += current_bins

        # print(new_df)
        # break
        data.triples = torch.cat((data.triples, new_df), 0)
    data.num_relations += 1
    data.num_entities += num_bins

    return data


def bin_numbers(data: Data, num_bins=3):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)

    i = 0
    cumsum = 0
    bins = np.arange(num_bins)

    for r in relevent_relations:
        current_bins = 0
        df_sub = data.triples[data.triples[:, 1] == r]
        # bins = np.digitize(sub_list,np.histogram(sub_list,num_bins,density = True)[1])
        sub_list = np.array([data.i2e[x][0]
                            for x in df_sub[:, 2]], dtype=np.float32)
        # mean = torch.round(torch.mean(df_sub[:,2],dtype = torch.float64)).to(torch.long)

        new = torch.tensor(df_sub[:, 0])

        new_df = torch.ones((new.size()[0], 3), dtype=torch.long)

        new_df[:, 0] = torch.tensor(df_sub[:, 0])  # values at place 0 (iri)
        new_df[:, 1] = data.num_relations + i
        binned_list = torch.from_numpy(np.digitize(
            sub_list, np.histogram(sub_list, num_bins)[1]))
        bi = 0
        while len(np.unique(binned_list)) < num_bins - bi:

            # print(f'{len(np.unique(binned_list))} vs {num_bins-bi}')
            bi += 1
            binned_list = torch.from_numpy(np.digitize(
                sub_list, np.histogram(sub_list, num_bins - bi)[1]))

        current_bins = len(np.unique(binned_list))
        new_df[:, 2] = binned_list
        new_df[:, 2] += (data.num_entities + ((i - 1) * num_bins)) - 1

        data.i2r.append(
            f"https://master-thesis.com/relations#sanity-check-{i}")
        data.r2i[f'https://master-thesis.com/relations#sanity-check-{i}'] = data.num_relations + i

        for e in range(current_bins):
            data.i2e.append(
                (f'https://master-thesis.com/entitys#sanity-check-target-{i}-{e}', 'preprocessed'))
            data.e2i[(
                f'https://master-thesis.com/entitys#sanity-check-target-{i}-{e}', 'preprocessed')] = data.num_entities + cumsum + e

        # i += 1
        # cumsum += current_bins

        # print(new_df)
        # break
        data.triples = torch.cat((data.triples, new_df), 0)
    data.num_relations += 1
    data.num_entities += num_bins

    return data


def bin_numbers2(data: Data, num_bins=3,use_lof=False):
    relevent_relations = get_relevant_relations(
        get_p_types(data), relevant_types=RDF_NUMBER_TYPES,r2i=data.r2i)

    i = 0
    cumsum = 0
    bins = np.arange(num_bins)

    for rr in relevent_relations:
        df = data.triples.clone()
        df = df[df[:, 1]== rr]
        sub_df = encode_number_sublist(df, data.i2e)
        if(use_lof):
            lof = LocalOutlierFactor(n_neighbors=10)
            lof.fit(sub_df[:,1].reshape(-1, 1))
            outlier_scores = lof.negative_outlier_factor_

            # Create a new column in the numpy array to store the outlier scores
            #tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
            threshold = np.percentile(outlier_scores, 10)
            # use the outlier scores to filter out the outliers from the numpy array
            sub_df = sub_df[outlier_scores > threshold]

        # numpy is used here since torch.histc was not working for some reason.
        sub_df = torch.cat( #put bins and sub_df together
            (sub_df, torch.from_numpy( #get numpy solutions back
                np.digitize( # assign for each value in sub_df the corresponding bin
                    sub_df[:, 1], np.histogram( # calculate n bins based on values in sub_df
                        sub_df[:, 1], num_bins)[1]
                )
            ).reshape(-1, 1) # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
            ), 1)

        for row in sub_df:
            s = data.i2e[row[0]]
            p = f'{URI_PREFIX}predicat#binning{rr}'
            o = (f'{URI_PREFIX}entity#binning{row[2]}',f'{URI_PREFIX}datatype#bin')
            add_triple(data,s,p,o)
    return data


    for r in relevent_relations:
        current_bins = 0
        df_sub = data.triples[data.triples[:, 1] == r]
        # bins = np.digitize(sub_list,np.histogram(sub_list,num_bins,density = True)[1])
        sub_list = np.array([data.i2e[x][0]
                            for x in df_sub[:, 2]], dtype=np.float32)
        # mean = torch.round(torch.mean(df_sub[:,2],dtype = torch.float64)).to(torch.long)

        new = torch.tensor(df_sub[:, 0])

        new_df = torch.ones((new.size()[0], 3), dtype=torch.long)

        new_df[:, 0] = torch.tensor(df_sub[:, 0])  # values at place 0 (iri)
        new_df[:, 1] = data.num_relations + i
        binned_list = torch.from_numpy(np.digitize(
            sub_list, np.histogram(sub_list, num_bins)[1]))
        bi = 0
        while len(np.unique(binned_list)) < num_bins - bi:

            # print(f'{len(np.unique(binned_list))} vs {num_bins-bi}')
            bi += 1
            binned_list = torch.from_numpy(np.digitize(
                sub_list, np.histogram(sub_list, num_bins - bi)[1]))

        current_bins = len(np.unique(binned_list))
        new_df[:, 2] = binned_list
        new_df[:, 2] += (data.num_entities + ((i - 1) * num_bins)) - 1

        data.i2r.append(
            f"https://master-thesis.com/relations#sanity-check-{i}")
        data.r2i[f'https://master-thesis.com/relations#sanity-check-{i}'] = data.num_relations + i

        for e in range(current_bins):
            data.i2e.append(
                (f'https://master-thesis.com/entitys#sanity-check-target-{i}-{e}', 'preprocessed'))
            data.e2i[(
                f'https://master-thesis.com/entitys#sanity-check-target-{i}-{e}', 'preprocessed')] = data.num_entities + cumsum + e

        # i += 1
        # cumsum += current_bins

        # print(new_df)
        # break
        data.triples = torch.cat((data.triples, new_df), 0)
    data.num_relations += 1
    data.num_entities += num_bins


def remove_outlier():
    pass


def encode_number_sublist(df: torch.Tensor, i2e: List[str]) -> torch.Tensor:
    sub_df = df.clone()
    for i in range(len(sub_df)):
        sub_df[i, 1] = torch.tensor(
            float(i2e[sub_df[i, 2]][0]), dtype=torch.float32)
    sub_df = sub_df[:, :2]
    return sub_df




def lof():
    pass

def kl_divergence():
    pass

def find_subpopulations():
    pass