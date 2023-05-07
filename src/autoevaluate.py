from omegaconf import DictConfig, OmegaConf
import hydra
import dataload
import embed
import evaluate
from utils.data_utils import data_to_kg, extract_ents, update_dataset_name, ensure_data_symmetry
import preprocess
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import logging
import os
import numpy as np
import re
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from main import evaluate_approach
import torch

log = logging.getLogger(__name__)

def calc_df()-> pd.DataFrame:
    if len(os.listdir("data/predicted/"))==0:
        return pd.DataFrame(columns=["f1_macro","f1_micro","count","dataset","eval_method","augment","embedder"])
    df = pd.DataFrame(columns=["full_name","f1_macro","f1_micro"])
        
    dfs = []
    i = 0
    for entry in os.listdir('data/predicted/'):
        results = np.loadtxt(f'data/predicted/{entry}', dtype=np.int32, delimiter=',')
        i+=1
        dfs.append(pd.DataFrame(
            [[
                entry,
                f1_score(results[0],results[1],average='micro'),
                f1_score(results[0],results[1],average='macro'),
            ]],
            columns=["full_name","f1_macro","f1_micro"]
        ))


    df = pd.concat(dfs,ignore_index=True)
    df['base_name']= df['full_name'].str.extract(r'(.*)\$[0-9]+\.csv')
    df['count'] = df.groupby('base_name')["base_name"].transform("count")
    df = df.groupby('base_name').mean().round(3)
    df = df.reset_index()
    df['dataset']= df['base_name'].str.extract(r'(.*?)\+.*')
    df['eval_method']= df['base_name'].str.extract(r'.*\$([A-Z]+)')
    df['augment'] = df['base_name'].str.extract(r'\+(.*?)\$.*')
    #df['augment'] = [', '.join(map(str, l)) for l in df['steps']]
    df['embedder'] = df['base_name'].str.extract(r'\$(.*?)\$')
    df = df.drop(columns='base_name')



    return df

@hydra.main(version_base=None, config_path="../config", config_name="multiple")
def multiple(cfg: DictConfig) -> None:
    torch.cuda.empty_cache()
    log.info("Load auto evaluater...")
    df = calc_df()
  
    for dataload in cfg['schedule']['dataload']:
        for augment in cfg['schedule']['augment']:

            for embed in cfg['schedule']['embed']:

                cfg['pipeline']['dataload'] = dataload
                cfg['pipeline']['augment'] = augment
                cfg['pipeline']['embed'] = embed
                iterations = cfg['schedule']['iterations']
                if len(df[(df["dataset"] == dataload)
                   & (df['embedder'] == embed) & (df['augment']== augment)])> 0:
                    print(df[(df["dataset"] == dataload)
                    & (df['embedder'] == embed) & (df['augment']== augment)]['count'].values[0])
                    
                    iterations = iterations- df[(df["dataset"] == dataload)
                    & (df['embedder'] == embed) & (df['augment']== augment)]['count'].values[0]

                if iterations > 0:
                    for i in range(iterations):
                        log.info(f"processing {dataload} | {augment} | {embed} (Nr. {i}/{iterations})")             
                        evaluate_approach(cfg)
                else:
                    log.info(f"skipping {dataload} | {augment} | {embed} -> iterations reached.")



if __name__ == '__main__':
    multiple()
