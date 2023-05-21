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
# from kgbench.load import Data
from utils import Data
from typing import List, Sequence, Tuple

from kgbench import load, tic, toc, d
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from utils import URI_PREFIX
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import datetime


def simplistic_approach(data: Data, **kwargs):
    return data


def encode_number_sublist(df: torch.Tensor, i2e: List[Tuple[str, str]]) -> torch.Tensor:
    sub_df = df.clone()
    sub_df = sub_df.to(torch.float32)
    for i in range(len(df)):
        sub_df[i, 1] = torch.tensor(
            float(i2e[df[i, 2]][0]), dtype=torch.float32)
    sub_df = sub_df[:, :2]
    return sub_df


def bin_numbers(data: Data, num_bins=3, use_lof=False, num_bins_as_percent=False, equal_height_binning=False, **kwargs):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)

    bin_percent = num_bins / 100

    if f'{URI_PREFIX}predicat#prevBin' not in data.r2i:
        p = f'{URI_PREFIX}predicat#prevBin'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f'{URI_PREFIX}predicat#nextBin'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for r in relevent_relations:
        p = f'{URI_PREFIX}predicat#binning{r}'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        sub_df = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e)

        if num_bins_as_percent:
            num_bins = math.floor(len(sub_df[:, 1].unique()) * bin_percent)
            if num_bins < 1:
                num_bins = 1
            print(num_bins)

        for b in range(num_bins):
            o = (f'{URI_PREFIX}entity#binning{b+1}#relation{relation}',
                 f'{URI_PREFIX}datatype#bin')
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1
            if (f'{URI_PREFIX}entity#binning{b}#relation{relation}', f'{URI_PREFIX}datatype#bin') in data.e2i:
                data = add_triple(data, o, f'{URI_PREFIX}predicat#prevBin', (
                    f'{URI_PREFIX}entity#binning{b}#relation{relation}', f'{URI_PREFIX}datatype#bin'))
                data = add_triple(data, (f'{URI_PREFIX}entity#binning{b}#relation{relation}',
                                  f'{URI_PREFIX}datatype#bin'), f'{URI_PREFIX}predicat#nextBin', o)

        augmented_df = data.triples.clone()
        augmented_df = augmented_df[augmented_df[:, 1] == relation]

        if (use_lof):
            lof = LocalOutlierFactor(n_neighbors=200)
            lof.fit(sub_df[:, 1].to(torch.int).reshape(-1, 1))
            outlier_scores = lof.negative_outlier_factor_
            # Create a new column in the numpy array to store the outlier scores
            # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
            threshold = np.percentile(outlier_scores, 5)
            # use the outlier scores to filter out the outliers from the numpy array
            outliers = sub_df[(outlier_scores <= threshold)
                              & (outlier_scores < -1)]
            sub_df = sub_df[(outlier_scores > threshold)
                            | (outlier_scores >= -1)]
            outlier_df = augmented_df[(outlier_scores <= threshold) & (
                outlier_scores < -1)].clone()
            augmented_df = augmented_df[(
                outlier_scores > threshold) | (outlier_scores >= -1)]
            if len(outliers) > 0:

                data.i2r.append(
                    f'{URI_PREFIX}predicat#outlier-{relation}')
                data.r2i[f'{URI_PREFIX}predicat#outlier-{relation}'] = data.num_relations

                data.i2e.append(
                    (f'{URI_PREFIX}entitys#outlier-{relation}', f'{URI_PREFIX}outlier'))
                data.e2i[(f'{URI_PREFIX}entitys#outlier-{relation}',
                          f'{URI_PREFIX}outlier')] = data.num_entities

                data.num_relations += 1
                data.num_entities += 1

                object_mapping = np.vectorize(lambda t: data.e2i[(
                    f'{URI_PREFIX}entitys#outlier-{relation}', f'{URI_PREFIX}outlier')])

                predicat_mapping = np.vectorize(
                    lambda t: data.r2i[f'{URI_PREFIX}predicat#outlier-{relation}'])

                outlier_df[:, 1] = torch.tensor(
                    np.array([predicat_mapping(outliers[:, 0])]), dtype=torch.int32)
                outlier_df[:, 2] = torch.tensor(
                    np.array([object_mapping(outliers[:, 0])]), dtype=torch.int32)
                data.triples = torch.cat((data.triples, outlier_df), 0)

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
            f'{URI_PREFIX}entity#binning{int(t)}#relation{relation}', f'{URI_PREFIX}datatype#bin')])

        predicat_mapping = np.vectorize(
            lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}'])

        augmented_df[:, 1] = torch.tensor(
            np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
        augmented_df[:, 2] = torch.tensor(
            np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
        data.triples = torch.cat((data.triples, augmented_df), 0)

    return data


def bin_numbers_5(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=5, use_lof=False)


def bin_numbers_10(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=10, use_lof=False)


def bin_numbers_100(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=100, use_lof=False)


def bin_numbers_lof_5(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=5, use_lof=True)


def bin_numbers_lof_10(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=10, use_lof=True)


def bin_numbers_lof_100(data: Data, **kwargs):
    return bin_numbers(data=data, num_bins=100, use_lof=True)


def bin_numbers_percentage_1(data, **kwargs):
    return bin_numbers(data, num_bins=1, num_bins_as_percent=True)


def bin_numbers_percentage_5(data, **kwargs):
    return bin_numbers(data, num_bins=5, num_bins_as_percent=True)


def bin_numbers_percentage_10(data, **kwargs):
    return bin_numbers(data, num_bins=10, num_bins_as_percent=True)


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


def bin_numbers_hierarchically(data: Data, list_num_bins=[3, 10, 100], **kwargs):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)

    # bin_percent = num_bins/100

    if f'{URI_PREFIX}predicat#prevBin' not in data.r2i:
        p = f'{URI_PREFIX}predicat#prevBin'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f'{URI_PREFIX}predicat#nextBin'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    if f'{URI_PREFIX}predicat#downBin' not in data.r2i:
        p = f'{URI_PREFIX}predicat#downBin'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f'{URI_PREFIX}predicat#upBin'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for num_bins in list_num_bins:
        for r in relevent_relations:
            p = f'{URI_PREFIX}predicat#binning{r}-num_bins{num_bins}'
            new_id = len(data.i2r)
            data.r2i[p] = new_id
            data.i2r.append(p)
            data.num_relations += 1

    for relation in relevent_relations:
        for i in range(len(list_num_bins)):
            num_bins = list_num_bins[i]
            sub_df = encode_number_sublist(
                data.triples[data.triples[:, 1] == relation], data.i2e)

            for b in range(num_bins):
                o = (f'{URI_PREFIX}entity#binning{b+1}-relation{relation}-num_bins{num_bins}',
                     f'{URI_PREFIX}datatype#bin')
                new_id = len(data.i2e)
                data.e2i[o] = new_id
                data.i2e.append(o)
                data.num_entities += 1
                prev_iri = (
                    f'{URI_PREFIX}entity#binning{b}-relation{relation}-num_bins{num_bins}', f'{URI_PREFIX}datatype#bin')
                if prev_iri in data.e2i:
                    data = add_triple(
                        data, o, f'{URI_PREFIX}predicat#prevBin', prev_iri)
                    data = add_triple(
                        data, prev_iri, f'{URI_PREFIX}predicat#nextBin', o)
                if i > 0:
                    upper_bin_nr = (b // list_num_bins[i - 1]) + 1
                    upper_iri = (
                        f'{URI_PREFIX}entity#binning{upper_bin_nr}-relation{relation}-num_bins{list_num_bins[i-1]}', f'{URI_PREFIX}datatype#bin')
                    data = add_triple(
                        data, o, f'{URI_PREFIX}predicat#upBin', upper_iri)
                    data = add_triple(
                        data, upper_iri, f'{URI_PREFIX}predicat#downBin', o)

            augmented_df = data.triples.clone()
            augmented_df = augmented_df[augmented_df[:, 1] == relation]

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
                f'{URI_PREFIX}entity#binning{int(t)}-relation{relation}-num_bins{num_bins}', f'{URI_PREFIX}datatype#bin')])

            predicat_mapping = np.vectorize(
                lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}-num_bins{num_bins}'])

            augmented_df[:, 1] = torch.tensor(
                np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
            augmented_df[:, 2] = torch.tensor(
                np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
            data.triples = torch.cat((data.triples, augmented_df), 0)

    return data




def bin_numbers_hierarchically_5_10_100(data:Data, list_num_bins=[5, 10, 100], **kwargs) -> Data:
    return bin_numbers_hierarchically(data, list_num_bins=list_num_bins)

def bin_numbers_hierarchically_5lvl_binary(data:Data, list_num_bins=[1,2,4,8,16], **kwargs) -> Data:
    return bin_numbers_hierarchically(data, list_num_bins=list_num_bins)

def altering_bins(data: Data, num_bins=5, **kwargs):
    relevent_relations = get_relevant_relations(
        data, relevant_types=RDF_NUMBER_TYPES)

    # bin_percent = num_bins/100

    if f'{URI_PREFIX}predicat#prevBin' not in data.r2i:
        p = f'{URI_PREFIX}predicat#prevBin'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f'{URI_PREFIX}predicat#nextBin'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1


    for r in relevent_relations:
        p = f'{URI_PREFIX}predicat#binning{r}'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        sub_df = encode_number_sublist(
            data.triples[data.triples[:, 1] == relation], data.i2e)

        for b in range(num_bins):
            o = (f'{URI_PREFIX}entity#binning{b+1}-relation{relation}',
                    f'{URI_PREFIX}datatype#bin')
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1
            prev_iri = (
                f'{URI_PREFIX}entity#binning{b}-relation{relation}', f'{URI_PREFIX}datatype#bin')
            if prev_iri in data.e2i:
                data = add_triple(
                    data, o, f'{URI_PREFIX}predicat#prevBin', prev_iri)
                data = add_triple(
                    data, prev_iri, f'{URI_PREFIX}predicat#nextBin', o)


        augmented_df = data.triples.clone()
        augmented_df = augmented_df[augmented_df[:, 1] == relation]

        # numpy is used here since torch.histc was not working for some reason.
        sub_df = torch.cat(  # put bins and sub_df together
            (sub_df, torch.from_numpy(  # get numpy solutions back
                np.digitize(  # assign for each value in sub_df the corresponding bin
                    sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
                        sub_df[:, 1], num_bins+1)[1][:-1]
                )
            ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
            ), 1)

        alternating_bins = []
        rel_id = data.r2i[f'{URI_PREFIX}predicat#binning{relation}']
        for triple in sub_df:
            if triple[2]<= num_bins:
                alternating_bins.append([int(triple[0]), rel_id, data.e2i[(
            f'{URI_PREFIX}entity#binning{int(triple[2])}-relation{relation}', f'{URI_PREFIX}datatype#bin')]])
            if triple[2]>1:
                alternating_bins.append([int(triple[0]), rel_id, data.e2i[(
            f'{URI_PREFIX}entity#binning{int(triple[2])-1}-relation{relation}', f'{URI_PREFIX}datatype#bin')]])
        
        alternating_bins = torch.tensor(alternating_bins, dtype = torch.int32)

        object_mapping = np.vectorize(lambda t: data.e2i[(
            f'{URI_PREFIX}entity#binning{int(t)}-relation{relation}', f'{URI_PREFIX}datatype#bin')])

        predicat_mapping = np.vectorize(
            lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}'])

        # augmented_df[:, 1] = torch.tensor(
        #     np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
        # augmented_df[:, 2] = torch.tensor(
        #     np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
        
        # alternating_triples = []
        # for entry in augmented_df:
        #     if 
        #     for i in num_bins:
        #         if data.e2i[entry[2]] == f'{URI_PREFIX}entity#binning{i+1}-relation{relation}':
        #             alternating_triples  = entry
        #             if :

        #     if data.e2i[entry[2]] == f'{URI_PREFIX}entity#binning{1}-relation{relation}':
        #         alternating_triples.append()
        data.triples = torch.cat((data.triples, alternating_bins), 0)

    return data

# def altering_bins(data: Data, num_bins=3, **kwargs):
#     relevent_relations = get_relevant_relations(
#         data, relevant_types=RDF_NUMBER_TYPES)

#     for b in range(num_bins):
#         o = (f'{URI_PREFIX}entity#binning{b+1}#1', f'{URI_PREFIX}datatype#bin')
#         new_id = len(data.i2e)
#         data.e2i[o] = new_id
#         data.i2e.append(o)
#         data.num_entities += 1

#         o = (f'{URI_PREFIX}entity#binning{b}#2', f'{URI_PREFIX}datatype#bin')
#         new_id = len(data.i2e)
#         data.e2i[o] = new_id
#         data.i2e.append(o)
#         data.num_entities += 1

#     for r in relevent_relations:
#         p = f'{URI_PREFIX}predicat#binning{r}#1'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1
#         p = f'{URI_PREFIX}predicat#binning{r}#2'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1

#     for relation in relevent_relations:

#         sub_df1 = encode_number_sublist(
#             data.triples[data.triples[:, 1] == relation], data.i2e)
#         sub_df2 = encode_number_sublist(
#             data.triples[data.triples[:, 1] == relation], data.i2e)

#         # numpy is used here since torch.histc was not working for some reason.
#         sub_df1 = torch.cat(  # put bins and sub_df together
#             (sub_df1, torch.from_numpy(  # get numpy solutions back
#                 np.digitize(  # assign for each value in sub_df the corresponding bin
#                     sub_df1[:, 1], np.histogram(  # calculate n bins based on values in sub_df
#                         sub_df1[:, 1], num_bins * 2)[1][:-1:2]
#                 )
#             ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
#             ), 1)
#         sub_df2 = torch.cat(  # put bins and sub_df together
#             (sub_df2, torch.from_numpy(  # get numpy solutions back
#                 np.digitize(  # assign for each value in sub_df the corresponding bin
#                     sub_df2[:, 1], np.histogram(  # calculate n bins based on values in sub_df
#                         sub_df2[:, 1], num_bins * 2)[1][1:-1:2]
#                 )
#             ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
#             ), 1)

#         object_mapping1 = np.vectorize(lambda t: data.e2i[(
#             f'{URI_PREFIX}entity#binning{t}#1', f'{URI_PREFIX}datatype#bin')])

#         predicat_mapping1 = np.vectorize(
#             lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}#1'])

#         sub_df1[:, 1] = torch.tensor(
#             [predicat_mapping1(sub_df1[:, 2])], dtype=torch.int32)
#         sub_df1[:, 2] = torch.tensor(
#             [object_mapping1(sub_df1[:, 2])], dtype=torch.int32)
#         data.triples = torch.cat((data.triples, sub_df1), 0)

#         object_mapping2 = np.vectorize(lambda t: data.e2i[(
#             f'{URI_PREFIX}entity#binning{t}#2', f'{URI_PREFIX}datatype#bin')])

#         bin_upper_bound = np.vectorize(
#             lambda t: num_bins - 1 if t >= num_bins else t)

#         predicat_mapping2 = np.vectorize(
#             lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}#2'])

#         sub_df2[:, 1] = torch.tensor(
#             [predicat_mapping2(sub_df2[:, 2])], dtype=torch.int32)
#         sub_df2[:, 2] = torch.tensor(
#             [bin_upper_bound(sub_df2[:, 2])], dtype=torch.int32)
#         sub_df2[:, 2] = torch.tensor(
#             [object_mapping2(sub_df2[:, 2])], dtype=torch.int32)

#         data.triples = torch.cat((data.triples, sub_df2), 0)
#         data = delete_empty_bin_types(data, num_bins * 2)
#     return data

def alternating_bins_10(data:Data, num_bins=10, **kwargs) -> Data:
    return altering_bins(data, num_bins=num_bins)


def alternating_bins_100(data:Data, num_bins=100, **kwargs) -> Data:
    return altering_bins(data, num_bins=num_bins)

# def bin_numbers_percentage(data: Data, num_bins=3, use_lof=False, num_bins_as_percent=False, equal_height_binning=False, **kwargs):
#     relevent_relations = get_relevant_relations(
#         data, relevant_types=RDF_NUMBER_TYPES)
#     percent_of_objects = num_bins/100

#     relation_bin_map = {}

#     for relation in relevent_relations:
#         if num_bins_as_percent:
#             sub_df = encode_number_sublist(
#             data.triples[data.triples[:, 1] == relation], data.i2e)
#             relation_bin_map[relation] = math.floor(len(sub_df[:,1].unique())*percent_of_objects)
#         else:
#             relation_bin_map[relation] = num_bins
#     max_bin = max([x for x in relation_bin_map.values()])
#     num_bins = max_bin
#     print(f'max_bin_number')


#     print(num_bins)
#     for b in range(num_bins):
#         o = (f'{URI_PREFIX}entity#binning{b+1}', f'{URI_PREFIX}datatype#bin')
#         new_id = len(data.i2e)
#         data.e2i[o] = new_id
#         data.i2e.append(o)
#         data.num_entities += 1

#     for r in relevent_relations:
#         p = f'{URI_PREFIX}predicat#binning{r}'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1

#     for relation in relevent_relations:

#         sub_df = encode_number_sublist(
#             data.triples[data.triples[:, 1] == relation], data.i2e)

#         # TODO test new function
#         if (use_lof):
#             lof = LocalOutlierFactor(n_neighbors=10)
#             lof.fit(sub_df[:, 1].reshape(-1, 1))
#             outlier_scores = lof.negative_outlier_factor_
#             # Create a new column in the numpy array to store the outlier scores
#             # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
#             threshold = np.percentile(outlier_scores, 10)
#             # use the outlier scores to filter out the outliers from the numpy array
#             sub_df = sub_df[outlier_scores > threshold]

#         if(num_bins_as_percent):
#             num_bins = math.floor(len(sub_df[:,1].unique())*percent_of_objects)
#             print(f'percentage based bins {percent_of_objects*100}% of unique literals results in {num_bins} bins')

#         # numpy is used here since torch.histc was not working for some reason.
#         sub_df = torch.cat(  # put bins and sub_df together
#             (sub_df, torch.from_numpy(  # get numpy solutions back
#                 np.digitize(  # assign for each value in sub_df the corresponding bin
#                     sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
#                         sub_df[:, 1], num_bins)[1][:-1]
#                 )
#             ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
#             ), 1)

#         object_mapping = np.vectorize(lambda t: data.e2i[(
#             f'{URI_PREFIX}entity#binning{t}', f'{URI_PREFIX}datatype#bin')])

#         predicat_mapping = np.vectorize(
#             lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}'])

#         sub_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
#         sub_df[:, 2] = torch.tensor(np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
#         data.triples = torch.cat((data.triples, sub_df), 0)
#     data = delete_empty_bin_types(data,max_bin)
#     return data

# def bin_numbers_percentage_3(data, **kwargs):
#     return bin_numbers_percentage(data,  3,False,True,False)

# def bin_numbers_percentage_5(data, **kwargs):
#     return bin_numbers_percentage(data,  5,False,True,False)

# def bin_numbers_percentage_15(data, **kwargs):
#     return bin_numbers_percentage(data,  15,False,True,False)

# def bin_numbers_lof_percentage_3(data, **kwargs):
#     return bin_numbers_percentage(data,  3,True,True,False)

# def bin_numbers_lof_percentage_5(data, **kwargs):
#     return bin_numbers_percentage(data,  5,True,True,False)

# def bin_numbers_lof_percentage_15(data, **kwargs):
#     return bin_numbers_percentage(data,  15,True,True,False)


# def bin_numbers_v3(data: Data, num_bins=3, use_lof=False, num_bins_as_percent=False, equal_height_binning=False, **kwargs):
#     relevent_relations = get_relevant_relations(
#         data, relevant_types=RDF_NUMBER_TYPES)
#     print(num_bins)


#     for r in relevent_relations:
#         p = f'{URI_PREFIX}predicat#binning{r}'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1

#     for relation in relevent_relations:
#         for b in range(num_bins):
#             o = (f'{URI_PREFIX}entity#binning{b+1}#relation{relation}', f'{URI_PREFIX}datatype#bin')
#             new_id = len(data.i2e)
#             data.e2i[o] = new_id
#             data.i2e.append(o)
#             data.num_entities += 1

#         sub_df = encode_number_sublist(
#             data.triples[data.triples[:, 1] == relation], data.i2e)

#         # TODO test new function
#         if (use_lof):
#             lof = LocalOutlierFactor(n_neighbors=10)
#             lof.fit(sub_df[:, 1].reshape(-1, 1))
#             outlier_scores = lof.negative_outlier_factor_
#             # Create a new column in the numpy array to store the outlier scores
#             # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
#             threshold = np.percentile(outlier_scores, 10)
#             # use the outlier scores to filter out the outliers from the numpy array
#             sub_df = sub_df[outlier_scores > threshold]

#         # numpy is used here since torch.histc was not working for some reason.
#         sub_df = torch.cat(  # put bins and sub_df together
#             (sub_df, torch.from_numpy(  # get numpy solutions back
#                 np.digitize(  # assign for each value in sub_df the corresponding bin
#                     sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
#                         sub_df[:, 1], num_bins)[1][:-1]
#                 )
#             ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
#             ), 1)

#         object_mapping = np.vectorize(lambda t: data.e2i[(
#             f'{URI_PREFIX}entity#binning{t}#relation{relation}', f'{URI_PREFIX}datatype#bin')])

#         predicat_mapping = np.vectorize(
#             lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}'])

#         sub_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
#         sub_df[:, 2] = torch.tensor(np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
#         data.triples = torch.cat((data.triples, sub_df), 0)
#     data = delete_empty_bin_types(data,num_bins*len(relevent_relations))
#     return data

# def bin_numbers_lof_100_v3(data: Data, **kwargs):
#     return bin_numbers_v3(data=data, num_bins=100, use_lof=True)

# def bin_numbers_100_v3(data: Data, **kwargs):
#     return bin_numbers_v3(data=data, num_bins=100, use_lof=False)

# def bin_numbers_v4(data: Data, num_bins=3, use_lof=False, num_bins_as_percent=False, equal_height_binning=False, **kwargs):
#     relevent_relations = get_relevant_relations(
#         data, relevant_types=RDF_NUMBER_TYPES)
#     print(num_bins)

#     if f'{URI_PREFIX}predicat#prevBin' not in data.r2i:
#         p = f'{URI_PREFIX}predicat#prevBin'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1
#         p = f'{URI_PREFIX}predicat#nextBin'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1

#     for r in relevent_relations:
#         p = f'{URI_PREFIX}predicat#binning{r}'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1

#     for relation in relevent_relations:
#         for b in range(num_bins):
#             o = (f'{URI_PREFIX}entity#binning{b+1}#relation{relation}', f'{URI_PREFIX}datatype#bin')
#             new_id = len(data.i2e)
#             data.e2i[o] = new_id
#             data.i2e.append(o)
#             data.num_entities += 1
#             if (f'{URI_PREFIX}entity#binning{b}#relation{relation}', f'{URI_PREFIX}datatype#bin') in data.e2i:
#                 data = add_triple(data,o,f'{URI_PREFIX}predicat#prevBin',(f'{URI_PREFIX}entity#binning{b}#relation{relation}', f'{URI_PREFIX}datatype#bin'))
#                 data = add_triple(data,(f'{URI_PREFIX}entity#binning{b}#relation{relation}', f'{URI_PREFIX}datatype#bin'),f'{URI_PREFIX}predicat#nextBin',o)
#         sub_df = encode_number_sublist(
#             data.triples[data.triples[:, 1] == relation], data.i2e)

#         # TODO test new function
#         if (use_lof):
#             lof = LocalOutlierFactor(n_neighbors=10)
#             lof.fit(sub_df[:, 1].reshape(-1, 1))
#             outlier_scores = lof.negative_outlier_factor_
#             # Create a new column in the numpy array to store the outlier scores
#             # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
#             threshold = np.percentile(outlier_scores, 10)
#             # use the outlier scores to filter out the outliers from the numpy array
#             sub_df = sub_df[outlier_scores > threshold]

#         # numpy is used here since torch.histc was not working for some reason.
#         sub_df = torch.cat(  # put bins and sub_df together
#             (sub_df, torch.from_numpy(  # get numpy solutions back
#                 np.digitize(  # assign for each value in sub_df the corresponding bin
#                     sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
#                         sub_df[:, 1], num_bins)[1][:-1]
#                 )
#             ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
#             ), 1)

#         object_mapping = np.vectorize(lambda t: data.e2i[(
#             f'{URI_PREFIX}entity#binning{t}#relation{relation}', f'{URI_PREFIX}datatype#bin')])

#         predicat_mapping = np.vectorize(
#             lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}'])

#         sub_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
#         sub_df[:, 2] = torch.tensor(np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
#         data.triples = torch.cat((data.triples, sub_df), 0)
#     #data = delete_empty_bin_types(data,num_bins*len(relevent_relations))
#     return data

# def bin_numbers_100_v4(data: Data, **kwargs):
#     return bin_numbers_v4(data=data, num_bins=100, use_lof=False)


# def bin_numbers_v5(data: Data, num_bins=3, use_lof=False, num_bins_as_percent=False, equal_height_binning=False, **kwargs):
#     relevent_relations = get_relevant_relations(
#         data, relevant_types=RDF_NUMBER_TYPES)
#     print(num_bins)

#     if f'{URI_PREFIX}predicat#prevBin' not in data.r2i:
#         p = f'{URI_PREFIX}predicat#prevBin'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1
#         p = f'{URI_PREFIX}predicat#nextBin'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1

#     for r in relevent_relations:
#         p = f'{URI_PREFIX}predicat#binning{r}'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1
#     for b in range(num_bins):
#         o = (f'{URI_PREFIX}entity#binning{b+1}', f'{URI_PREFIX}datatype#bin')
#         new_id = len(data.i2e)
#         data.e2i[o] = new_id
#         data.i2e.append(o)
#         data.num_entities += 1
#         if (f'{URI_PREFIX}entity#binning{b}', f'{URI_PREFIX}datatype#bin') in data.e2i:
#             data = add_triple(data,o,f'{URI_PREFIX}predicat#prevBin',(f'{URI_PREFIX}entity#binning{b}', f'{URI_PREFIX}datatype#bin'))
#             data = add_triple(data,(f'{URI_PREFIX}entity#binning{b}', f'{URI_PREFIX}datatype#bin'),f'{URI_PREFIX}predicat#nextBin',o)

#     for relation in relevent_relations:

#         sub_df = encode_number_sublist(
#             data.triples[data.triples[:, 1] == relation], data.i2e)

#         # TODO test new function
#         if (use_lof):
#             lof = LocalOutlierFactor(n_neighbors=10)
#             lof.fit(sub_df[:, 1].reshape(-1, 1))
#             outlier_scores = lof.negative_outlier_factor_
#             # Create a new column in the numpy array to store the outlier scores
#             # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
#             threshold = np.percentile(outlier_scores, 10)
#             # use the outlier scores to filter out the outliers from the numpy array
#             sub_df = sub_df[outlier_scores > threshold]

#         # numpy is used here since torch.histc was not working for some reason.
#         sub_df = torch.cat(  # put bins and sub_df together
#             (sub_df, torch.from_numpy(  # get numpy solutions back
#                 np.digitize(  # assign for each value in sub_df the corresponding bin
#                     sub_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
#                         sub_df[:, 1], num_bins)[1][:-1]
#                 )
#             ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
#             ), 1)

#         object_mapping = np.vectorize(lambda t: data.e2i[(
#             f'{URI_PREFIX}entity#binning{t}', f'{URI_PREFIX}datatype#bin')])

#         predicat_mapping = np.vectorize(
#             lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{relation}'])

#         sub_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
#         sub_df[:, 2] = torch.tensor(np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
#         data.triples = torch.cat((data.triples, sub_df), 0)
#     #data = delete_empty_bin_types(data,num_bins*len(relevent_relations))
#     return data

# def bin_numbers_100_v5(data: Data, **kwargs):
#     return bin_numbers_v4(data=data, num_bins=100, use_lof=False)





def bin_on_subpopulations(data:Data, base_relation:int, strategy = 'relations', max_depth=2):
    #get triples
    triples = data.triples
    relevant_triples = triples[triples[:,1] == base_relation]
    
    s_list = [[relevant_triples[:,0].numpy()]] # list with subject for each level of tree
    #get all subjects possessing relation 'base_relation'
    df = pd.DataFrame(relevant_triples[:,0], columns=['s']) 
    ## add val column containing numerical values coresponding to the s base_relation o triple, then mapped according to i2e[o]
    df['val'] = df.s.apply(lambda s: int(float(data.i2e[relevant_triples[relevant_triples[:,0]==s][0,2]][0])))

    #calculate and sort most common relations among entities 
    r_counts = get_relations_for_entites(data, df.s)
    r_counts = r_counts[r_counts['r']!=base_relation] # here better logic is needed


    for i in range(1): # not yet implemented for deeper walks
        parent = s_list[0][0]
        # print(len(parent))
        c_triples = triples[(triples[:,1] == base_relation) & torch.isin(triples[:,0], torch.tensor(parent) )] #same as relavant triples
        # print(len(c_triples))
        # print(c_triples)
        #child_reminder = parent
        
        sub_s_list = [] 
        #found = False
        #j = 0

        for j in range(len(r_counts)):
            #c_triples = triples[(triples[:,1] == base_relation) & torch.isin(triples[:,0], child_reminder )]

            sub_triples = triples[(triples[:,1] == base_relation) & (torch.isin(triples[:,0], triples[triples[:,1] == r_counts.iloc[j].r][:,0])) & (torch.isin(triples[:,0], c_triples[:,0]))]
            proportion = len(sub_triples)/len(parent)
            kl_div = adapted_kl_divergence(df[df.s.isin(c_triples[:,0].numpy())], df[df.s.isin(sub_triples[:,0].numpy())])
            
            print(f'j: {j} - prop: {proportion} - kl_div: {kl_div}')

            if proportion>0.95 or proportion <0.05:
                pass
            #need to define value here
            elif kl_div<300:
                pass
            else:
                #print("was here")
                # print("added")
                sub_s_list.append(sub_triples[:,0].numpy())
                c_triples = c_triples[~torch.isin(c_triples[:,0],sub_triples[:,0])]

        sub_s_list.append(c_triples[:,0].numpy())
        if len(sub_s_list) >1:
            s_list.append(sub_s_list)
    return df, s_list


def subpopulation_binning(data:Data, num_bins:int, use_lof:bool)->Data:
    relevent_relations = get_relevant_relations(data, RDF_NUMBER_TYPES)
    for b in range(num_bins):
        o = (f'{URI_PREFIX}entity#binning{b+1}', f'{URI_PREFIX}datatype#bin')
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1

    for rr in relevent_relations:
        df, s_list = bin_on_subpopulations(data,rr)
        if len(s_list)>1:
            print(f'{rr} has {len(s_list[1])} subpopulations')
            for i in range(len(s_list[1])):
                p = f'{URI_PREFIX}predicat#binning{rr}#subpopulation{i}'
                new_id = len(data.i2r)
                data.r2i[p] = new_id
                data.i2r.append(p)
                data.num_relations += 1
            
            sub_df = encode_number_sublist(
            data.triples[data.triples[:, 1] == rr], data.i2e)
        
            for i in range(len(s_list[1])):
            
                sub_population_df = sub_df[torch.isin(sub_df[:,0],torch.Tensor(s_list[1][i]))].clone()
                print(f'len subpopulation {len(sub_population_df)}')

                if (use_lof):
                    lof = LocalOutlierFactor(n_neighbors=10)
                    lof.fit(sub_population_df[:, 1].reshape(-1, 1))
                    outlier_scores = lof.negative_outlier_factor_
                    # Create a new column in the numpy array to store the outlier scores
                    # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
                    threshold = np.percentile(outlier_scores, 10)
                    # use the outlier scores to filter out the outliers from the numpy array
                    sub_population_df = sub_population_df[outlier_scores > threshold]
                sub_population_df = torch.cat(  # put bins and sub_df together
                    (sub_population_df, torch.from_numpy(  # get numpy solutions back
                        np.digitize(  # assign for each value in sub_df the corresponding bin
                        sub_population_df[:, 1], np.histogram(  # calculate n bins based on values in sub_df
                            sub_population_df[:, 1], num_bins)[1][:-1]
                            )
                        ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
                    ), 1)

                object_mapping = np.vectorize(lambda t: data.e2i[(
                    f'{URI_PREFIX}entity#binning{t}', f'{URI_PREFIX}datatype#bin')])

                predicat_mapping = np.vectorize(
                    lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{rr}#subpopulation{i}'])

                sub_population_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_population_df[:, 2])]), dtype=torch.int32)
                sub_population_df[:, 2] = torch.tensor(np.array([object_mapping(sub_population_df[:, 2])]), dtype=torch.int32)
                data.triples = torch.cat((data.triples, sub_population_df), 0)
            else:
                p = f'{URI_PREFIX}predicat#binning{rr}'
                new_id = len(data.i2r)
                data.r2i[p] = new_id
                data.i2r.append(p)
                data.num_relations += 1
            
                sub_df = encode_number_sublist(
                data.triples[data.triples[:, 1] == rr], data.i2e)
                if (use_lof):
                    lof = LocalOutlierFactor(n_neighbors=10)
                    lof.fit(sub_df[:, 1].reshape(-1, 1))
                    outlier_scores = lof.negative_outlier_factor_
                    # Create a new column in the numpy array to store the outlier scores
                    # tensor_np = torch.hstack((encoded_df, outlier_scores.reshape(-1,1)))
                    threshold = np.percentile(outlier_scores, 10)
                    # use the outlier scores to filter out the outliers from the numpy array
                    sub_df = sub_df[outlier_scores > threshold]

            

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
                    lambda t: data.r2i[f'{URI_PREFIX}predicat#binning{rr}'])

                sub_df[:, 1] = torch.tensor(np.array([predicat_mapping(sub_df[:, 2])]), dtype=torch.int32)
                sub_df[:, 2] = torch.tensor(np.array([object_mapping(sub_df[:, 2])]), dtype=torch.int32)
                data.triples = torch.cat((data.triples, sub_df), 0)


            
    data = delete_empty_bin_types(data,num_bins)
    return data

def get_enteties_for_relation_with_entites(data:Data,entities, r):
    df = data.triples
    df = df[(torch.isin(df[:,0], torch.tensor(entities))) & (df[:,1]==r)]
    o, counts = torch.unique(df[:,2], return_counts=True)
    return pd.DataFrame({'o':o.numpy(),'count':counts.numpy()}).sort_values('count',ascending=False).reset_index(drop=True)

def adapted_kl_divergence(p_dist_a,p_dist_b, num_bins = 10):
    a_bin = np.histogram(p_dist_a.val,num_bins)[0]
    b_bin = np.histogram(p_dist_b.val,num_bins)[0]
    a_len = len(p_dist_a.val)
    b_len = len(p_dist_b.val)
    sum_l =0
    e = 0.1**10
    for i in range(len(a_bin)):
        sum_l += np.log((a_bin[i]/(b_bin[i]+e))+e)*a_bin[i]
    return round(b_len/a_len * sum_l , 5)