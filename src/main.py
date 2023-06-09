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
import sys
import numpy as np
import torch
import random
log = logging.getLogger(__name__)
BASEPATH = "data/preprocessed"
import pickle
from kgbench.load import Data


@hydra.main(version_base=None, config_path="../config", config_name="single")
def evaluate_one(cfg:DictConfig)->None:
    evaluate_approach(cfg)

def load_preprocessed_data(cfg: DictConfig)->Data:
    with open(f'{BASEPATH}/{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}.pickle', "rb") as f:
        data:Data = pickle.load(f)
    return data
def save_preprocessed_data(data:Data, cfg: DictConfig):
    with open(f'{BASEPATH}/{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}.pickle', "wb") as f:
        pickle.dump(data, f)
def exist_preprocessed_data(cfg: DictConfig):
    return os.path.exists(f'{BASEPATH}/{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}.pickle')



    

def evaluate_approach(cfg: DictConfig) -> None:
    np.random.seed(42) 
    torch.manual_seed(42)
    random.seed(42)
    os.environ['PYTHONHASHSEED'] = '42'
    #os.execv(sys.executable, [sys.executable] + sys.argv)

    log.info("Data loading...")
    if exist_preprocessed_data(cfg):
        data = load_preprocessed_data(cfg)
        log.info("Preprocessed Data found, Skipping Preprocess...")
    else:
        data = getattr(dataload, cfg["pipeline"]["dataload"])(**cfg["dataload"][cfg["pipeline"]["dataload"]])
        data.name = f'{cfg["pipeline"]["dataload"]}+{cfg["pipeline"]["augment"]}'
    # data = update_dataset_name(
    #     data, cfg["preprocess"], cfg["pipeline"]["preprocess"])
    
        log.info("Preprocess started...")
        for step in cfg['aug_approach'][cfg["pipeline"]["augment"]]:
            log.info(f"Processing step {step}...")
            data = getattr(preprocess, step)(data, **cfg["aug_method"][step])
        save_preprocessed_data(data,cfg)


    log.info("Embedding started...")
    embedder = getattr(embed, cfg["pipeline"]["embed"])(data,
                                                        **cfg["embed"][cfg["pipeline"]["embed"]])

    train_entities, test_entities, train_target, test_taget = extract_ents(
        data)  # extract necessary fields from data

    log.info("fit_transform")
    embeddings, train_embeddings, test_embeddings = embedder.fit_transform()
    version = 0 # if approach allready ambedded
    embeddings_base_path = f'{cfg["file_paths"]["embedded"]}/{data.name}${cfg["pipeline"]["embed"]}$'
    while(os.path.exists(f'{embeddings_base_path}train${str(version)}.csv')):
        version +=1
    
    np.savetxt(f'{embeddings_base_path}train${str(version)}.csv',train_embeddings,delimiter=',',fmt="%s")
    np.savetxt(f'{embeddings_base_path}test${str(version)}.csv',test_embeddings,delimiter=',',fmt="%s")

    # TODO pickle embedder

    log.info("Classifier fitting started...")
    # TODO pack into 1 step or create map instead of list to be able to know model name.
    models = {}
    for m in cfg["pipeline"]["evaluate"]:
        log.info(f'fitting {m}...')
        model = getattr(evaluate, m)(
            **cfg["evaluate"][m])
        model.fit(train_embeddings, train_target)
        models[m] = model
        
    # if embeddings.dtype == "complex64":
    #     train_embeddings = train_embeddings.real
    #     test_embeddings = test_embeddings.real
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
    evaluate_one()
