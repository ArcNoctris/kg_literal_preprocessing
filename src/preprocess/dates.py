from utils import get_relevant_relations, get_p_types, RDF_DATE_TYPES, add_triple, URI_PREFIX
import pandas as pd
from utils import Data,  RDF_DATE_TYPES
import datetime
import numpy as np
# from preprocess.binning import delete_empty_bin_types

from utils import add_triple
import torch


# # def append_weekday_and_month(data:Data,**kwargs) -> Data:
# #     df = data.triples
# #     p_types = get_p_types(data)
# #     rr = get_relevant_relations(p_types, RDF_DATE_TYPES, data.r2i)
# #     for r in rr:
# #         dfs = []
# #         for triple in df[df[:,1] ==r]:
# #             dfs.append(pd.DataFrame([[int(triple[0]), data.i2e[triple[2]][0].split('+')[0]]], columns = ["s","date"]))
# #         frame = pd.concat(dfs)
# #         frame.date = frame.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') if type(x)==str else np.NaN)
# #         frame["weekday"] = frame.date.apply(lambda x: x.strftime('%A') if x != np.NaN else np.NaN)
# #         frame["month"] = frame.date.apply(lambda x: x.strftime('%B') if x != np.NaN else np.NaN)
# #         for _, row in frame.iterrows():
# #             s = data.i2e[row['s']]
# #             p_week = f'{URI_PREFIX}predicat#dates-weekday-{r}'
# #             o_week = (f'{URI_PREFIX}entity#dates-weekday-{row["weekday"]}',f'{URI_PREFIX}datatype#weekday')
# #             p_month = f'{URI_PREFIX}predicat#dates-month-{r}'
# #             o_month = (f'{URI_PREFIX}entity#dates-month-{row["month"]}',f'{URI_PREFIX}datatype#month')
# #             data = add_triple(data,s,p_week,o_week)
# #             data = add_triple(data,s,p_month,o_month)
# #     return data


# def append_date_features(data:Data,**kwargs) -> Data:


#     date_features =  ['day_of_month','day_of_week','month_of_year','quarter_of_year','year']
#     feature_ranges = {
#     # 'day_of_month' : ["01","02","03","04","05","06","07","08","09","10","11","12", "13","14","15","16","17","18","19","20","21","22","23","24", "01","02","03","04","05","06","07","08","09","10","11","12"]
#     'day_of_week' : ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
#     'month_of_year' : ["01","02","03","04","05","06","07","08","09","10","11","12"],
#     'quarter_of_year' : ['1',"2","3","4"]
#     }


#     # df = data.triples
#     relevent_relations = get_relevant_relations(data, RDF_DATE_TYPES)

#     if f'{URI_PREFIX}predicat#prevDate' not in data.r2i:
#         p = f'{URI_PREFIX}predicat#prevDate'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1

#         p = f'{URI_PREFIX}predicat#nextDate'
#         new_id = len(data.i2r)
#         data.r2i[p] = new_id
#         data.i2r.append(p)
#         data.num_relations += 1

#     for r in relevent_relations:
#         for feature in date_features:
#             p = f'{URI_PREFIX}predicat#aug_date_{r}_{feature}'
#             new_id = len(data.i2r)
#             data.r2i[p] = new_id
#             data.i2r.append(p)
#             data.num_relations += 1

#     for relation in relevent_relations:
#         for feature in date_features:
#             if feature in feature_ranges:
#                 for i in range(len(feature_ranges[feature])):
#                     entry = feature_ranges[feature][i]
#                     o = (f'{URI_PREFIX}entity#{feature}{entry}-relation{relation}',
#                         f'{URI_PREFIX}datatype#feature')
#                     new_id = len(data.i2e)
#                     data.e2i[o] = new_id
#                     data.i2e.append(o)
#                     data.num_entities += 1
#                     if i > 0:
#                         prev_o = (f'{URI_PREFIX}entity#{feature}{feature_ranges[feature][i-1]}-relation{relation}',
#                         f'{URI_PREFIX}datatype#feature')
#                         data = add_triple(data, o, f'{URI_PREFIX}predicat#prevDate', prev_o)
#                         data = add_triple(data, prev_o, f'{URI_PREFIX}predicat#nextDate', o)

#         df = pd.DataFrame(data.triples[data.triples[:,1]== relation], columns = ["s","p","o"])
#         df['t'] = df['o'].apply(lambda x: data.i2e[x][0])
#         df['t']= pd.to_datetime(df['t'],errors='coerce')
#         df = df[df['t'].notnull()]
#         df['day_of_month'] = df['t'].apply(lambda x: str(x.strftime('%d')) if not pd.isnull(x) else "")
#         df['day_of_week'] = df['t'].apply(lambda x: str(x.strftime('%A')) if not pd.isnull(x) else "")
#         df['month_of_year'] = df['t'].apply(lambda x: str(x.strftime('%m')) if not pd.isnull(x) else "")
#         df['quarter_of_year'] = df['t'].apply(lambda x: str(((int(x.strftime('%m'))-1)//4)+1) if not pd.isnull(x) else "")
#         df['year'] = df['t'].apply(lambda x: str(x.strftime('%Y')) if not pd.isnull(x) else "")

#         for feature in date_features:
#             if feature not in feature_ranges:
#                 for f in df[df[feature].notnull()][feature].unique():
#                     o = (f'{URI_PREFIX}entity#{feature}{f}-relation{relation}',
#                     f'{URI_PREFIX}datatype#feature')
#                     new_id = len(data.i2e)
#                     data.e2i[o] = new_id
#                     data.i2e.append(o)
#                     data.num_entities += 1

#         for feature in date_features:
#             # if feature in feature_ranges:
                
#             df['new_o'] = df[feature].apply(lambda f: data.e2i[(f'{URI_PREFIX}entity#{feature}{f}-relation{relation}',
#                     f'{URI_PREFIX}datatype#feature')] if (not pd.isnull(f))&(f!="")  else np.nan)
#             df['new_p'] = df[feature].apply(lambda f: data.r2i[f'{URI_PREFIX}predicat#aug_date_{relation}_{feature}'] if (not pd.isnull(f))&(f!="") else np.nan)
#             #print(df)
#             ten = torch.tensor(df[(df['new_o'].notnull())][['s','new_p','new_o']].values.astype(np.int32), dtype= torch.int32)
#             data.triples = torch.cat((data.triples, ten), 0)

#     # data = delete_empty_bin_types(data,num_topics)
#     return data

def append_date_features(data:Data,**kwargs) -> Data:


    date_features =  ['day_of_month','day_of_week','month_of_year','quarter_of_year','year']
    feature_ranges = {
    'day_of_week' : ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
    'month_of_year' : ["01","02","03","04","05","06","07","08","09","10","11","12"],
    'quarter_of_year' : ['1',"2","3","4"]
    }

    relevent_relations = get_relevant_relations(data, RDF_DATE_TYPES)

    if f'{URI_PREFIX}predicat#prevDate' not in data.r2i:
        p = f'{URI_PREFIX}predicat#prevDate'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f'{URI_PREFIX}predicat#nextDate'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        for feature in date_features:
            p = f'{URI_PREFIX}predicat#aug_date_{relation}_{feature}'
            new_id = len(data.i2r)
            data.r2i[p] = new_id
            data.i2r.append(p)
            data.num_relations += 1

    for relation in relevent_relations:

        df = pd.DataFrame(data.triples[data.triples[:,1]== relation], columns = ["s","p","o"])
        df['t'] = df['o'].apply(lambda x: data.i2e[x][0])
        df['t']= pd.to_datetime(df['t'],errors='coerce')
        df = df[df['t'].notnull()]
        df['day_of_month'] = df['t'].apply(lambda x: str(x.strftime('%d')))
        df['day_of_week'] = df['t'].apply(lambda x: str(x.strftime('%A')) )
        df['month_of_year'] = df['t'].apply(lambda x: str(x.strftime('%m')))
        df['quarter_of_year'] = df['t'].apply(lambda x: str(((int(x.strftime('%m'))-1)//4)+1))
        df['year'] = df['t'].apply(lambda x: str(x.strftime('%Y')))

        for feature in date_features:
            if feature in feature_ranges:
                for i in range(len(feature_ranges[feature])):
                    entry = feature_ranges[feature][i]
                    o = (f'{URI_PREFIX}entity#{feature}{entry}-relation{relation}',
                        f'{URI_PREFIX}datatype#feature')
                    new_id = len(data.i2e)
                    data.e2i[o] = new_id
                    data.i2e.append(o)
                    data.num_entities += 1
                    if i > 0:
                        prev_o = (f'{URI_PREFIX}entity#{feature}{feature_ranges[feature][i-1]}-relation{relation}',
                        f'{URI_PREFIX}datatype#feature')
                        data = add_triple(data, o, f'{URI_PREFIX}predicat#prevDate', prev_o)
                        data = add_triple(data, prev_o, f'{URI_PREFIX}predicat#nextDate', o)
            else:
                for f in df[feature].unique():
                    o = (f'{URI_PREFIX}entity#{feature}{f}-relation{relation}',
                    f'{URI_PREFIX}datatype#feature')
                    new_id = len(data.i2e)
                    data.e2i[o] = new_id
                    data.i2e.append(o)
                    data.num_entities += 1

        for feature in date_features:
            # if feature in feature_ranges:
                
            df['new_o'] = df[feature].apply(lambda f: data.e2i[(f'{URI_PREFIX}entity#{feature}{f}-relation{relation}',
                    f'{URI_PREFIX}datatype#feature')] if (not pd.isnull(f))&(f!="")  else np.nan)
            df['new_p'] = df[feature].apply(lambda f: data.r2i[f'{URI_PREFIX}predicat#aug_date_{relation}_{feature}'] if (not pd.isnull(f))&(f!="") else np.nan)
            #print(df)
            ten = torch.tensor(df[(df['new_o'].notnull())][['s','new_p','new_o']].values.astype(np.int32), dtype= torch.int32)
            data.triples = torch.cat((data.triples, ten), 0)

    # data = delete_empty_bin_types(data,num_topics)
    return data



def bin_dates(data:Data,num_bins=100,**kwargs) -> Data:
    relevent_relations = get_relevant_relations(data, RDF_DATE_TYPES)

    if f'{URI_PREFIX}predicat#prevDate' not in data.r2i:
        p = f'{URI_PREFIX}predicat#prevDate'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

        p = f'{URI_PREFIX}predicat#nextDate'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:
        p = f'{URI_PREFIX}predicat#bin_date_{relation}'
        new_id = len(data.i2r)
        data.r2i[p] = new_id
        data.i2r.append(p)
        data.num_relations += 1

    for relation in relevent_relations:

        for i in range(num_bins):
            # entry = feature_ranges[feature][i]
            o = (f'{URI_PREFIX}entity#{i+1}-relation{relation}',
                f'{URI_PREFIX}datatype#bin')
            new_id = len(data.i2e)
            data.e2i[o] = new_id
            data.i2e.append(o)
            data.num_entities += 1
            if i > 0:
                prev_o = (f'{URI_PREFIX}entity#{i}-relation{relation}',
                f'{URI_PREFIX}datatype#bin')
                data = add_triple(data, o, f'{URI_PREFIX}predicat#prevDate', prev_o)
                data = add_triple(data, prev_o, f'{URI_PREFIX}predicat#nextDate', o)

        df = pd.DataFrame(data.triples[data.triples[:,1]== relation], columns = ["s","p","o"])
        df['t'] = df['o'].apply(lambda x: data.i2e[x][0])
        df['t']= pd.to_datetime(df['t'],errors='coerce')
        df = df[df['t'].notnull()]
        df['t'] = df['t'].values.astype("int")

        df['bins'] = torch.from_numpy(  # get numpy solutions back
                    np.digitize(  # assign for each value in sub_df the corresponding bin
                        df['t'].values, np.histogram(  # calculate n bins based on values in sub_df
                            df['t'].values, num_bins)[1][:-1]
                    )
                ).reshape(-1, 1)  # transfrom x tensor into (x,1) tensor to fit (x,2) shape of sub_df
            
        
        df['new_o'] = df['bins'].apply(lambda x: data.e2i[(f'{URI_PREFIX}entity#{x}-relation{relation}',
                f'{URI_PREFIX}datatype#bin')])
        df['new_p'] = df['bins'].apply(lambda x: data.r2i[f'{URI_PREFIX}predicat#bin_date_{relation}'])
        ten = torch.tensor(df[(df['new_o'].notnull())][['s','new_p','new_o']].values.astype(np.int32), dtype= torch.int32)
        data.triples = torch.cat((data.triples, ten), 0)
    return data



#def append_month():
#    pass