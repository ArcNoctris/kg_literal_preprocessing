from omegaconf import DictConfig, OmegaConf
import hydra
import dataload
import embed
import evaluate
from utils.data_utils import data_to_kg, extract_ents, update_dataset_name
import preprocess
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import logging
import os
import numpy as np
import re
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

log = logging.getLogger(__name__)

def calc_df()-> pd.DataFrame:
    df = pd.DataFrame(columns=["full_name","f1_macro","f1_micro"])
        
    dfs = []
    for entry in os.listdir('data/predicted/'):
        results = np.loadtxt(f'data/predicted/{entry}', dtype=np.int32, delimiter=',')
    
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
    df = df.groupby('base_name').mean().round(3)
    #df['name'] = df.index
    df = df.reset_index()
    df['dataset']= df['base_name'].str.extract(r'(.*?)\+.*')
    df['eval_method']= df['base_name'].str.extract(r'.*\$([A-Z]+)')
    df['steps'] = df['base_name'].str.findall(r'\+(.*?)-')
    df['steps'] = [', '.join(map(str, l)) for l in df['steps']]
    df['embedder'] = df['base_name'].str.extract(r'\$(.*?)\$')
    df = df.drop(columns='base_name')
    return df

@hydra.main(version_base=None, config_path="../config", config_name="main")
def process_n_times(cfg: DictConfig) -> None:
    log.info("Load auto evaluater...")
    df = calc_df()
    for dataload in cfg['pipeline_schedule']['dataload']:
        for preprocess in cfg['pipeline_schedule']['preprocess']:
            preprocess_string =""
            for i in range(len(preprocess)):
                preprocess_string = preprocess_string + preprocess[i]
                if i < len(preprocess)-1:
                    preprocess_string = preprocess_string +", "
            for embed in cfg['pipeline_schedule']['embed']:
                iterations = cfg['pipeline_schedule']['iterations'] - len(df[(df["dataset"] == dataload)
                  & (df['embedder'] == embed) & (df['steps']== preprocess_string)])
                
                cfg['pipeline']['dataload'] = dataload
                cfg['pipeline']['preprocess'] = preprocess
                cfg['pipeline']['embed'] = embed

                if iterations == 0:
                    log.info(f"skipping {dataload} | {preprocess} | {embed} -> iterations reached.")
                if iterations > 0:
                    for i in range(iterations):

                        log.info(f"processing {dataload} | {preprocess} | {embed} (Nr. {i}/{iterations})")             

                        process(cfg)
 


def process(cfg: DictConfig) -> None:


    log.info("Data loading...")
    # TODO only load if preprocessed file not available
    print(cfg["preprocess"][cfg["pipeline"]["preprocess"][0]])
    data = getattr(dataload, cfg["pipeline"]["dataload"])(**cfg["dataload"][cfg["pipeline"]["dataload"]])
    data.name = cfg["pipeline"]["dataload"]
    data = update_dataset_name(
        data, cfg["preprocess"], cfg["pipeline"]["preprocess"])

    log.info("Preprocess started...")
    for step in cfg["pipeline"]["preprocess"]:
        log.info(f"Processing step {step}...")
        data = getattr(preprocess, step)(data, **cfg["preprocess"][step])

    # TODO save preprocessed file 

    log.info("Embedding started...")
    # TODO load embedder if allready there ? (not best idea since i want to do 5 seperate embeddings for each embedding method)
    embedder = getattr(embed, cfg["pipeline"]["embed"])(data,
                                                        **cfg["embed"][cfg["pipeline"]["embed"]])

    train_entities, test_entities, train_target, test_taget = extract_ents(
        data)  # extract necessary fields from data

    log.info("fit_transform")
    embeddings, train_embeddings, test_embeddings = embedder.fit_transform()
    version = 0
    embeddings_base_path = f'{cfg["file_paths"]["embedded"]}/{data.name}${cfg["pipeline"]["embed"]}$'
    while(os.path.exists(f'{embeddings_base_path}train${str(version)}.csv')):
        version +=1
    
    np.savetxt(f'{embeddings_base_path}train${str(version)}.csv',train_embeddings,delimiter=',',fmt="%s")
    np.savetxt(f'{embeddings_base_path}test${str(version)}.csv',test_embeddings,delimiter=',',fmt="%s")

    # TODO pickle embedder
    # TODO save embeddings and be able to save multiple of same embedding method (e.g. _0, _1 ...)
    # prio 1
    log.info("Classifier fitting started...")
    # TODO pack into 1 step or create map instead of list to be able to know model name.
    models = {}
    for m in cfg["pipeline"]["evaluate"]:
        log.info(f'fitting {m}...')
        model = getattr(evaluate, m)(
            **cfg["evaluate"][m])
        model.fit(train_embeddings, train_target)
        models[m] = model

    log.info("Evaluation started...")
    # TODO save into file to analyze in subsequent stages (also with _0,_1 and so on)
    # prio 1.1
    for m, model in models.items():
        log.info(f"evaluating model {m}")
        predictions = model.predict(test_embeddings)
        version = 0
        predictions_base_path = f'{cfg["file_paths"]["predicted"]}/{data.name}${cfg["pipeline"]["embed"]}${m}'
        while(os.path.exists(f'{predictions_base_path}${str(version)}.csv')):
            version +=1
        np.savetxt(f'{predictions_base_path}${str(version)}.csv',[predictions, test_taget],delimiter=',',fmt="%s")

        log.info(
            f"Predicted {len(test_entities)} entities with an accuracy of "
            + f"{accuracy_score(test_taget, predictions) * 100 :.4f}%"
        )
        log.info(f'resulted in following f scores: micro {f1_score(test_taget, predictions, average="micro")} macro {f1_score(test_taget, predictions, average="macro")}')
        log.info("Confusion Matrix :")
        log.info(confusion_matrix(test_taget, predictions))



    log.info("Save Data...")

if __name__ == '__main__':
    process_n_times()
