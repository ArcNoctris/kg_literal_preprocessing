from omegaconf import DictConfig, OmegaConf
import hydra
import dataload
import embed
import evaluate
from utils.data_utils import data_to_kg, extract_ents
import preprocess
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def process(cfg: DictConfig) -> None:


    log.info("Evaluation started...")
    model = getattr(evaluate, cfg["pipeline"]["evaluate"])(
        **cfg["evaluate"][cfg["pipeline"]["evaluate"]])



if __name__ == '__main__':
    process()
