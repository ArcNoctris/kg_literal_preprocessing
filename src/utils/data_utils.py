from pyrdf2vec.graphs import KG
from pyrdf2vec.graphs import KG, Vertex
# from kgbench.load import Data
from utils import Data
from utils import RDF_NUMBER_TYPES
from typing import List, Dict, Tuple
import torch


def update_dataset_name(data: Data, preprocess_args, preprocess_steps) -> Data:
    #data.name = data.name + "+"
    for i in range(len(preprocess_steps)):
        if i < len(preprocess_steps):
            data.name = data.name + "+"
        data.name = data.name + f"{preprocess_steps[i]}"
        keys=list(preprocess_args[preprocess_steps[i]].keys())

        for j in range(len(keys)):
            if j < len(keys):
                data.name = data.name + "-"
            data.name = data.name + f"{keys[j]}@{str(preprocess_args[preprocess_steps[i]][keys[j]])}"
    return data

    

def ensure_data_symmetry(data: Data)-> Data:
    # REMAP to symmetric relations a <-> b
    #maped_data = data.triples.copy
    for t in data.triples:
        t[0] = torch.tensor(data.e2i[data.i2e[t[0]]], dtype=torch.int32)
        t[1] = torch.tensor(data.r2i[data.i2r[t[1]]], dtype=torch.int32)
        t[2] = torch.tensor(data.e2i[data.i2e[t[2]]], dtype=torch.int32)

    for t in data.training:
        t[0] = torch.tensor(data.e2i[data.i2e[t[0]]])

    for t in data.withheld:
        t[0] = torch.tensor(data.e2i[data.i2e[t[0]]])
    base_e_unique = torch.unique(torch.cat([data.triples[:,0],data.triples[:,2]])) 
    base_r_unique = torch.unique(data.triples[:,1])

    new_e2i = {}
    new_i2e = []

    # TODO filter problem possibly here?!?! todo tomorrow
    
    for i in range(len(data.i2e)):
        if i in base_e_unique.numpy():
        #print("here")
            new_e2i[data.i2e[i]] = len(new_i2e)
            new_i2e.append(data.i2e[i])

    #create new r mapping
    new_r2i = {}
    new_i2r = []

    for i in range(len(data.i2r)):
        if i in base_r_unique.numpy():
            new_r2i[data.i2r[i]] = len(new_i2r)
            new_i2r.append(data.i2r[i])

    for t in data.triples:
        t[0] = torch.tensor(new_e2i[data.i2e[t[0]]], dtype=torch.int32)
        t[1] = torch.tensor(new_r2i[data.i2r[t[1]]], dtype=torch.int32)
        t[2] = torch.tensor(new_e2i[data.i2e[t[2]]], dtype=torch.int32)

    for t in data.training:
        t[0] = torch.tensor(new_e2i[data.i2e[t[0]]])

    for t in data.withheld:
        t[0] = torch.tensor(new_e2i[data.i2e[t[0]]])

    #update metedata
    data.num_entities = len(new_i2e)
    data.num_relations = len(new_i2r)

    #     #update data
    #data.triples = filtered
    data.i2e = new_i2e
    data.e2i = new_e2i
    data.i2r = new_i2r
    data.r2i = new_r2i
    return data


def extract_ents(data: Data):
    train_entities = []
    train_target = []
    for d in data.training:
        ent = data.i2e[d[0]][0]
        train_entities.append(ent)
        train_target.append(int(d[1]))

    test_entities = []
    test_taget = []
    for d in data.withheld:
        ent = data.i2e[d[0]][0]
        test_entities.append(ent)
        test_taget.append(int(d[1]))

    entities = train_entities + test_entities
    target = train_target + test_taget
    return train_entities, test_entities, train_target, test_taget


def data_to_kg(data: Data):
    kg = KG()
    # kg.literals = literals_preds
    for triple in data.triples:
        # subject_type = data.i2e[triple[0]][1]
        # object_type = data.i2e[triple[2]][1]
        # if subject_type == "iri" and object_type == "iri":

        subj = Vertex(*[data.i2e[triple[0]][0]])
        obj = Vertex(*[data.i2e[triple[2]][0]])
        pred = Vertex(*[data.i2r[triple[1]]],
                      **{"predicate":True, "vprev":subj, "vnext":obj})
        kg.add_walk(subj, pred, obj)
    return kg


def get_relevant_relationssss(data: Data, rdf_type=RDF_NUMBER_TYPES) -> List[int]:
    object_types = {}
    for triple in data.triples:
        object_name = data.i2r[triple[1]]
        subject_type = data.i2e[triple[0]][1]
        predicat_type = data.i2e[triple[2]][1]
        if object_name not in object_types:
            object_types[object_name] = ([], [])
        if subject_type not in object_types[object_name][0]:
            object_types[object_name][0].append(subject_type)
        if predicat_type not in object_types[object_name][1]:
            object_types[object_name][1].append(predicat_type)

    relevent_relations: List[int] = []
    for ot in object_types.keys():
        for nt in rdf_type:
            if nt in object_types[ot][1]:
                if data.r2i[ot] not in relevent_relations:
                    relevent_relations.append(data.r2i[ot])
    return relevent_relations

# Will get private after some tests are done


def get_p_types(data: Data) -> Dict[str, Tuple[List[str], List[str]]]:
    p_types = {}
    for triple in data.triples:
        o_type = data.i2e[triple[0]][1]
        s_type = data.i2e[triple[2]][1]
        p = data.i2r[triple[1]]

        if p not in p_types:
            p_types[p] = ([o_type], [s_type])
        else:
            if o_type not in p_types[p][0]:
                p_types[p][0].append(o_type)
            if s_type not in p_types[p][1]:
                p_types[p][1].append(s_type)
    return p_types


def get_relevant_relations(data: Data, relevant_types: List[str]) -> List[int]:
    p_types = get_p_types(data)
    relevent_relations: List[int] = []
    for ptk, ptv in p_types.items():
        for nt in relevant_types:
            if nt in ptv[1] and data.r2i[ptk] not in relevent_relations:
                relevent_relations.append(data.r2i[ptk])
    return relevent_relations


def add_triple(data: Data, s: Tuple[str, str], p: str, o: Tuple[str, str], verbose=0) -> Data:
    if s not in data.i2e:
        #print(f'{s}')
        #print(f'{p}')
        #print(f'{o}')
        #print(data.e2i[s])
        new_id = len(data.i2e)
        data.e2i[s] = new_id
        data.i2e.append(s)
        data.num_entities += 1
        if (verbose > 0):
            print(f'created new entity:')
            print(f'{data.e2i[s]} - {s}')
    if o not in data.i2e:
        new_id = len(data.i2e)
        data.e2i[o] = new_id
        data.i2e.append(o)
        data.num_entities += 1
        if (verbose > 0):
            print(f'created new entity:')
            print(f'{data.e2i[o]} - {o}')

    if p not in data.i2r:
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1
        if (verbose > 0):
            print(f'created new relation:')
            print(f'{data.r2i[p]} - {p}')
    si = data.e2i[s]
    pi = data.r2i[p]
    oi = data.e2i[o]
    if (verbose > 1):
        print(f'added triple:')
        print(f'{si} - {pi} - {oi}')
    new_triple = torch.tensor([[si, pi, oi]], dtype=torch.int32)
    data.triples = torch.cat((data.triples, new_triple), 0)
    return data


# def delete_triple(data: Data, si, pi, oi, verbose=0) -> Data:

#     triples = data.triples
#     triples = triples[~((triples[:, 0] == si) & (
#         triples[:, 1] == pi) & (triples[:, 2] == oi))]
#     data.triples = triples
#     to_pop = []

#     if len(triples[triples[:, 0] == si]) == 0 and len(triples[triples[:, 2] == si]) == 0:
#         to_pop.append(int(si.numpy()))

#     if len(triples[triples[:, 1] == pi]) == 0:
#         data.i2r.pop(int(pi.numpy()))
#         updated_r2i:Dict[str,int] = {}
#         for i in data.i2r:
#             updated_r2i[data.i2r[i]] = i
#         data.r2i = updated_r2i
#         data.num_relations -= 1
#         if (verbose > 0):
#             print(f'deleted relation:')
#             print(f'{pi}')

#     if len(triples[triples[:, 0] == oi]) == 0 and len(triples[triples[:, 2] == oi]) == 0:
#         to_pop.append(int(oi.numpy()))
#     to_pop.sort(reverse=True)
#     if len(to_pop) > 0:
#         for elem in to_pop:
#             data.i2e.pop(elem)
#         updated_e2i = {}
#         for i in range(len(data.i2e)):
#             updated_e2i[data.i2e[i]] = i
#         data.e2i = updated_e2i
#         data.num_entities -= 1
#         if (verbose > 0):
#             print(f'deleted entity:')
#             print(f'{to_pop}')

#     return data

def delete_r(data:Data, r):
    # get subset data
    filtered = data.triples[~(torch.isin(data.triples[:,1],r))]
    # get neg e filter
    base_e_unique = torch.unique(torch.cat([data.triples[:,0],data.triples[:,2]])) 
    filtered_e_unique = torch.unique(torch.cat([filtered[:,0],filtered[:,2]])) 
    neg_e_filter =  base_e_unique[~(torch.isin(base_e_unique, filtered_e_unique))]

    #get neg r filter
    base_r_unique = torch.unique(data.triples[:,1])
    filtered_r_unique = torch.unique(filtered[:,1])
    neg_r_filter =  base_r_unique[~(torch.isin(base_r_unique, filtered_r_unique))]

    #create new e mapping
    new_e2i = {}
    new_i2e = []


    # TODO filter problem possibly here?!?! todo tomorrow
    
    for i in range(len(data.i2e)):
        if i not in neg_e_filter.numpy():
            new_e2i[data.i2e[i]] = len(new_i2e)
            new_i2e.append(data.i2e[i])

    #create new r mapping
    new_r2i = {}
    new_i2r = []

    for i in range(len(data.i2r)):
        if i not in neg_r_filter.numpy():
            new_r2i[data.i2r[i]] = len(new_i2r)
            new_i2r.append(data.i2r[i])

    # apply new mapping for triples
    for t in filtered:
        t[0] = new_e2i[data.i2e[t[0]]]
        t[1] = new_r2i[data.i2r[t[1]]]
        t[2] = new_e2i[data.i2e[t[2]]]
        #t[0] = torch.tensor(new_e2i[data.i2e[t[0].numpy()]], dtype=torch.int32)
        #t[1] = torch.tensor(new_r2i[data.i2r[t[1].numpy()]], dtype=torch.int32)
        #t[2] = torch.tensor(new_e2i[data.i2e[t[2].numpy()]], dtype=torch.int32)

    # create new train & withheld
    new_train =  []
    new_withheld = []

    # calculate new train & withheld
    for ent in data.training:
        new_train.append([new_e2i[data.i2e[ent[0].numpy()]],ent[1]])
    
    for ent in data.withheld:
        new_withheld.append([new_e2i[data.i2e[ent[0].numpy()]],ent[1]])

    #update metedata
    data.num_entities = len(new_i2e)
    data.num_relations = len(new_i2r)

    #update data
    data.triples = filtered
    data.i2e = new_i2e
    data.e2i = new_e2i
    data.i2r = new_i2r
    data.r2i = new_r2i
    data.training = torch.tensor(new_train)
    data.withheld = torch.tensor(new_withheld)
    
    return data

