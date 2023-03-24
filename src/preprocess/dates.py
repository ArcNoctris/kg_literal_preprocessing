from utils import get_relevant_relations, get_p_types, RDF_DATE_TYPES, add_triple, URI_PREFIX
import pandas as pd
from kgbench.load import Data
import datetime
import numpy as np



def append_weekday_and_month(data:Data,**kwargs) -> Data:
    df = data.triples
    p_types = get_p_types(data)
    rr = get_relevant_relations(p_types, RDF_DATE_TYPES, data.r2i)
    for r in rr:
        dfs = []
        for triple in df[df[:,1] ==r]:
            dfs.append(pd.DataFrame([[int(triple[0]), data.i2e[triple[2]][0].split('+')[0]]], columns = ["s","date"]))
        frame = pd.concat(dfs)
        frame.date = frame.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S') if type(x)==str else np.NaN)
        frame["weekday"] = frame.date.apply(lambda x: x.strftime('%A') if x != np.NaN else np.NaN)
        frame["month"] = frame.date.apply(lambda x: x.strftime('%B') if x != np.NaN else np.NaN)
        for _, row in frame.iterrows():
            s = data.i2e[row['s']]
            p_week = f'{URI_PREFIX}predicat#dates-weekday-{r}'
            o_week = (f'{URI_PREFIX}entity#dates-weekday-{row["weekday"]}',f'{URI_PREFIX}datatype#weekday')
            p_month = f'{URI_PREFIX}predicat#dates-month-{r}'
            o_month = (f'{URI_PREFIX}entity#dates-month-{row["month"]}',f'{URI_PREFIX}datatype#month')
            data = add_triple(data,s,p_week,o_week)
            data = add_triple(data,s,p_month,o_month)
    return data




#def append_month():
#    pass