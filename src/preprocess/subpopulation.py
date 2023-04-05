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

def get_enteties_for_relation_with_entites(data:Data,entities, r):
    df = data.triples
    df = df[(torch.isin(df[:,0], torch.tensor(entities))) & (df[:,1]==r)]
    o, counts = torch.unique(df[:,2], return_counts=True)
    return pd.DataFrame({'o':o.numpy(),'count':counts.numpy()}).sort_values('count',ascending=False).reset_index(drop=True)

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

def get_relations_for_entites(data:Data,entities):
    df = data.triples
    df = df[torch.isin(df[:,0], torch.tensor(entities))]
    r, counts = torch.unique(df[:,1], return_counts=True)
    return pd.DataFrame({'r':r.numpy(),'count':counts.numpy()}).sort_values('count',ascending=False).reset_index(drop=True)

def bin_on_subpopulations2(data:Data, base_relation:int):
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
    o_counts = []
    for r in r_counts.r.array:
        o_count = get_enteties_for_relation_with_entites(data,df.s,r)
        if o_count['count'].iloc[0]/o_count['count'].sum() > 0.05 and len(o_count['count'])>1 and o_count['count'].sum()/ df.s.count() > 0.05:
            o_counts.append((r,o_count))

    parent = s_list[0][0]
    # print(len(parent))
    c_triples = triples[(triples[:,1] == base_relation) & torch.isin(triples[:,0], torch.tensor(parent) )] #same as relavant triples
    # print(len(c_triples))
    # print(c_triples)
        #child_reminder = parent
        
    sub_s_list = []
      
    for i in range(len(o_counts)):
        for j in range(len(o_counts[i])):    
            #c_triples = triples[(triples[:,1] == base_relation) & torch.isin(triples[:,0], child_reminder )]

            sub_triples = triples[(triples[:,1] == base_relation) & (torch.isin(triples[:,0], triples[triples[:,1] == o_counts[i][0]][:,0])) & (torch.isin(triples[:,0], c_triples[:,0])) & (torch.isin(triples[:,0], triples[triples[:,2] == o_counts[i][1].iloc[j].o][:,0]))]
           
           
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
            # print(o_counts[i][0])
                sub_s_list.append(sub_triples[:,0].numpy())
                c_triples = c_triples[~torch.isin(c_triples[:,0],sub_triples[:,0])]

    sub_s_list.append(c_triples[:,0].numpy())
    if len(sub_s_list) >1:
        s_list.append(sub_s_list)
    return df, s_list

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


def bound_subpopulation_binning(data:Data, num_bins:int, use_lof:bool)->Data:
    relevent_relations = get_relevant_relations(data, RDF_NUMBER_TYPES)
    for b in range(num_bins):
        o = (f'{URI_PREFIX}entity#binning{b+1}', f'{URI_PREFIX}datatype#bin')
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1

    for rr in relevent_relations:
        df, s_list = bin_on_subpopulations2(data,rr)
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

def subpopulation_binning_lof_100(data:Data, **kwargs):
    return subpopulation_binning(data, 100, True)

def subpopulation_binning_lof_10(data:Data, **kwargs):
    return subpopulation_binning(data, 10, True)

def subpopulation_binning_10(data:Data, **kwargs):
    return subpopulation_binning(data, 10, False)

def bound_subpopulation_binning_lof_10(data:Data, **kwargs):
    return bound_subpopulation_binning(data, 10, True)

def bound_subpopulation_binning_lof_100(data:Data, **kwargs):
    return bound_subpopulation_binning(data, 100, True)

def bound_subpopulation_binning_10(data:Data, **kwargs):
    return bound_subpopulation_binning(data, 10, False)