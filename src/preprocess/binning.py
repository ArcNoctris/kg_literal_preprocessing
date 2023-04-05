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
from utils import RDF_NUMBER_TYPES, get_relevant_relations, add_triple, get_p_types, ALL_LITERALS
from kgbench.load import Data
from typing import List, Sequence, Tuple

from kgbench import load, tic, toc, d
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils import URI_PREFIX
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import datetime


def do_nothing(data: Data, **kwargs):
    return data
def simplistic_approach(data: Data, **kwargs):
    return data

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

        data.triples = torch.cat((data.triples, new_df), 0)
    data.num_relations += 1
    data.num_entities += num_bins

    return data


def bin_numbers2(data: Data, num_bins=3, use_lof=True, **kwargs):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)

    i = 0
    cumsum = 0
    bins = np.arange(num_bins)

    for rr in relevent_relations:
        df = data.triples.clone()
        df = df[df[:, 1] == rr]
        sub_df = encode_number_sublist(df, data.i2e)
        if (use_lof):
            lof = LocalOutlierFactor(n_neighbors=10)
            lof.fit(sub_df[:, 1].reshape(-1, 1))
            outlier_scores = lof.negative_outlier_factor_

            # Create a new column in the numpy array to store the outlier scores
            # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
            threshold = np.percentile(outlier_scores, 10)
            # use the outlier scores to filter out the outliers from the numpy array
            sub_df = sub_df[outlier_scores > threshold]

        # numpy is used here since torch.histc was not working for some reason.
        sub_df = torch.cat(  # put bins and sub_df together
            (sub_df, torch.from_numpy(  # get numpy solutions back
                np.digitize(  # assign for each value in sub_df the corresponding bin
                    sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
                        sub_df[:, 1], num_bins)[1]
                )
            ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
            ), 1)

        for row in sub_df:
            s = data.i2e[row[0]]
            p = f'{URI_PREFIX}predicat#binning{rr}'
            o = (
                f'{URI_PREFIX}entity#binning{row[2]}', f'{URI_PREFIX}datatype#bin')
            data = add_triple(data, s, p, o)
    return data



def encode_number_sublist(df: torch.Tensor, i2e: List[str]) -> torch.Tensor:
    sub_df = df.clone()
    for i in range(len(sub_df)):
        sub_df[i, 1] = torch.tensor(
            float(i2e[sub_df[i, 2]][0]), dtype=torch.float32)
    sub_df = sub_df[:, :2]
    return sub_df


def one_entity(data: Data, **kwargs):
    rr = get_relevant_relations(data, ALL_LITERALS)
    for r in rr:
        df = data.triples[data.triples[:, 1] == r]
        new_df = df.clone().detach()

        new_df[:, 1] = data.num_relations
        new_df[:, 2] = data.num_entities

        data.i2r.append(
            f"https://master-thesis.com/relations#one-relation-{r}")
        data.r2i[f'https://master-thesis.com/relations#one-relation-{r}'] = data.num_relations

        data.i2e.append(
            (f'https://master-thesis.com/entitys#one-literal-{r}', 'preprocessed'))
        data.e2i[(
            f'https://master-thesis.com/entitys#one-literal-{r}', 'preprocessed')] = data.num_entities

        data.triples = torch.cat((data.triples, new_df), 0)
        data.num_relations += 1
        data.num_entities += 1
    return data

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


def bin_numbers(data: Data, num_bins=3, use_lof=False, num_bins_as_percent=False, equal_height_binning=False, **kwargs):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)
    print(num_bins)
    for b in range(num_bins):
        o = (f'{URI_PREFIX}entity#binning{b+1}', f'{URI_PREFIX}datatype#bin')
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1

    for r in relevent_relations:
        p = f'{URI_PREFIX}predicat#binning{r}'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:

        sub_df = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e)

        # TODO test new function
        if (use_lof):
            lof = LocalOutlierFactor(n_neighbors=10)
            lof.fit(sub_df[:, 1].reshape(-1, 1))
            outlier_scores = lof.negative_outlier_factor_
            # Create a new column in the numpy array to store the outlier scores
            # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
            threshold = np.percentile(outlier_scores, 10)
            # use the outlier scores to filter out the outliers from the numpy array
            sub_df = sub_df[outlier_scores > threshold]

        # numpy is used here since torch.histc was not working for some reason.
        sub_df = torch.cat(  # put bins and sub_df together
            (sub_df, torch.from_numpy(  # get numpy solutions back
                np.digitize(  # assign for each value in sub_df the corresponding bin
                    sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
                        sub_df[:, 1], num_bins)[1][:-1]
                )
            ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
            ), 1)

        object_mapping = np.vectorize(lambda t: data.e2i[(
            f'{URI_PREFIX}entity#binning{t}', f'{URI_PREFIX}datatype#bin')])

        predicat_mapping = np.vectorize(
            lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}'])

        sub_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
        sub_df[:, 2] = torch.tensor(np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
        data.triples = torch.cat((data.triples, sub_df), 0)
    data = delete_empty_bin_types(data,num_bins)
    return data


def bin_numbers_3(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=3, use_lof=False)


def bin_numbers_10(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=10, use_lof=False)


def bin_numbers_100(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=100, use_lof=False)


def bin_numbers_lof_3(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=3, use_lof=True)


def bin_numbers_lof_10(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=10, use_lof=True)


def bin_numbers_lof_100(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=100, use_lof=True)

def bin_numbers_hierarchically(data, list_num_bins=[3,10,100], **kwargs):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)
    #print(num_bins)
    for num_bins in list_num_bins:
        for b in range(num_bins):
            o = (f'{URI_PREFIX}entity#binning{b+1}#num_bins{num_bins}', f'{URI_PREFIX}datatype#bin')
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1

    for num_bins in list_num_bins:
        for r in relevent_relations:
            p = f'{URI_PREFIX}predicat#binning{r}#num_bins{num_bins}'
            new_id = len(data.i2r)
            data.r2i[p] = new_id
            data.i2r.append(p)
            data.num_relations += 1
    for num_bins in list_num_bins:
        for relation in relevent_relations:

            sub_df = encode_number_sublist(
                data.triples[data.triples[:, 1] == relation], data.i2e)

        # # TODO test new function
        # if (use_lof):
        #     lof = LocalOutlierFactor(n_neighbors=10)
        #     lof.fit(sub_df[:, 1].reshape(-1, 1))
        #     outlier_scores = lof.negative_outlier_factor_
        #     # Create a new column in the numpy array to store the outlier scores
        #     # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
        #     threshold = np.percentile(outlier_scores, 10)
        #     # use the outlier scores to filter out the outliers from the numpy array
        #     sub_df = sub_df[outlier_scores > threshold]

            # numpy is used here since torch.histc was not working for some reason.
            sub_df = torch.cat(  # put bins and sub_df together
                (sub_df, torch.from_numpy(  # get numpy solutions back
                    np.digitize(  # assign for each value in sub_df the corresponding bin
                        sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
                            sub_df[:, 1], num_bins)[1][:-1]
                    )
                ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
                ), 1)

            object_mapping = np.vectorize(lambda t: data.e2i[(
                f'{URI_PREFIX}entity#binning{t}#num_bins{num_bins}', f'{URI_PREFIX}datatype#bin')])

            predicat_mapping = np.vectorize(
                lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}#num_bins{num_bins}'])

            sub_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
            sub_df[:, 2] = torch.tensor(np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
            data.triples = torch.cat((data.triples, sub_df), 0)
    data = delete_empty_bin_types(data,np.sum(list_num_bins))
    return data

def bin_numbers_hierarchically_3_10_100(data, list_num_bins=[3,10,100], **kwargs):
    return bin_numbers_hierarchically(data, list_num_bins=list_num_bins)

def altering_bins(data: Data, num_bins=3, **kwargs):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)

    for b in range(num_bins):
        o = (f'{URI_PREFIX}entity#binning{b+1}#1', f'{URI_PREFIX}datatype#bin')
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1

        o = (f'{URI_PREFIX}entity#binning{b}#2', f'{URI_PREFIX}datatype#bin')
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1

    for r in relevent_relations:
        p = f'{URI_PREFIX}predicat#binning{r}#1'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1
        p = f'{URI_PREFIX}predicat#binning{r}#2'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:

        sub_df1 = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e)
        sub_df2 = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e)

        # numpy is used here since torch.histc was not working for some reason.
        sub_df1 = torch.cat(  # put bins and sub_df together
            (sub_df1, torch.from_numpy(  # get numpy solutions back
                np.digitize(  # assign for each value in sub_df the corresponding bin
                    sub_df1[:, 1], np.histogram(  # calculate n bins based on values in sub_df
                        sub_df1[:, 1], num_bins*2)[1][:-1:2]
                )
            ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
            ), 1)
        sub_df2 = torch.cat(  # put bins and sub_df together
            (sub_df2, torch.from_numpy(  # get numpy solutions back
                np.digitize(  # assign for each value in sub_df the corresponding bin
                    sub_df2[:, 1], np.histogram(  # calculate n bins based on values in sub_df
                        sub_df2[:, 1], num_bins*2)[1][1:-1:2]
                )
            ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
            ), 1)

        object_mapping1 = np.vectorize(lambda t: data.e2i[(
            f'{URI_PREFIX}entity#binning{t}#1', f'{URI_PREFIX}datatype#bin')])

        predicat_mapping1 = np.vectorize(
            lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}#1'])

        sub_df1[:, 1] = torch.tensor([predicat_mapping1(sub_df1[:, 2])], dtype=torch.int32)
        sub_df1[:, 2] = torch.tensor([object_mapping1(sub_df1[:, 2])], dtype=torch.int32)
        data.triples = torch.cat((data.triples, sub_df1), 0)

        object_mapping2 = np.vectorize(lambda t: data.e2i[(
            f'{URI_PREFIX}entity#binning{t}#2', f'{URI_PREFIX}datatype#bin')])
        
        bin_upper_bound = np.vectorize(lambda t: num_bins-1 if t>=num_bins else t)

        predicat_mapping2 = np.vectorize(
            lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}#2'])

        sub_df2[:, 1] = torch.tensor([predicat_mapping2(sub_df2[:, 2])], dtype=torch.int32)
        sub_df2[:, 2] = torch.tensor([bin_upper_bound(sub_df2[:, 2])], dtype=torch.int32)
        sub_df2[:, 2] = torch.tensor([object_mapping2(sub_df2[:, 2])], dtype=torch.int32)
        
        data.triples = torch.cat((data.triples, sub_df2), 0)
        data = delete_empty_bin_types(data,num_bins*2)
    return data



def bin_numbers_percentage(data: Data, num_bins=3, use_lof=False, num_bins_as_percent=False, equal_height_binning=False, **kwargs):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)
    percent_of_objects = num_bins/100
    
    relation_bin_map = {}

    for relation in relevent_relations:
        if num_bins_as_percent:
            sub_df = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e)
            relation_bin_map[relation] = math.floor(len(sub_df[:,1].unique())*percent_of_objects)
        else:
            relation_bin_map[relation] = num_bins
    max_bin = max([x for x in relation_bin_map.values()])
    num_bins = max_bin
    print(f'max_bin_number')


    print(num_bins)
    for b in range(num_bins):
        o = (f'{URI_PREFIX}entity#binning{b+1}', f'{URI_PREFIX}datatype#bin')
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1

    for r in relevent_relations:
        p = f'{URI_PREFIX}predicat#binning{r}'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:

        sub_df = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e)

        # TODO test new function
        if (use_lof):
            lof = LocalOutlierFactor(n_neighbors=10)
            lof.fit(sub_df[:, 1].reshape(-1, 1))
            outlier_scores = lof.negative_outlier_factor_
            # Create a new column in the numpy array to store the outlier scores
            # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
            threshold = np.percentile(outlier_scores, 10)
            # use the outlier scores to filter out the outliers from the numpy array
            sub_df = sub_df[outlier_scores > threshold]

        if(num_bins_as_percent):
            num_bins = math.floor(len(sub_df[:,1].unique())*percent_of_objects)
            print(f'percentage based bins {percent_of_objects*100}% of unique literals results in {num_bins} bins')

        # numpy is used here since torch.histc was not working for some reason.
        sub_df = torch.cat(  # put bins and sub_df together
            (sub_df, torch.from_numpy(  # get numpy solutions back
                np.digitize(  # assign for each value in sub_df the corresponding bin
                    sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
                        sub_df[:, 1], num_bins)[1][:-1]
                )
            ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
            ), 1)

        object_mapping = np.vectorize(lambda t: data.e2i[(
            f'{URI_PREFIX}entity#binning{t}', f'{URI_PREFIX}datatype#bin')])

        predicat_mapping = np.vectorize(
            lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}'])

        sub_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
        sub_df[:, 2] = torch.tensor(np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
        data.triples = torch.cat((data.triples, sub_df), 0)
    data = delete_empty_bin_types(data,max_bin)
    return data

def bin_numbers_percentage_3(data, **kwargs):
    return bin_numbers_percentage(data,  3,False,True,False)

def bin_numbers_percentage_5(data, **kwargs):
    return bin_numbers_percentage(data,  5,False,True,False)

def bin_numbers_percentage_15(data, **kwargs):
    return bin_numbers_percentage(data,  15,False,True,False)

def bin_numbers_lof_percentage_3(data, **kwargs):
    return bin_numbers_percentage(data,  3,True,True,False)

def bin_numbers_lof_percentage_5(data, **kwargs):
    return bin_numbers_percentage(data,  5,True,True,False)

def bin_numbers_lof_percentage_15(data, **kwargs):
    return bin_numbers_percentage(data,  15,True,True,False)