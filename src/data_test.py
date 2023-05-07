from omegaconf import DictConfig, OmegaConf
import dataload
import hydra
import datetime

# Get the current time
start = datetime.datetime.now()

# Print the current time
# print("start time", start)
# data = dataload.dmg777k()
# end = datetime.datetime.now()
# print("end_time", end)
# print("elapsed:",end-start)
#print(data.i2e)
@hydra.main(version_base=None, config_path="../config", config_name="multiple")
def sth(cfg:DictConfig):
    #a = hydra.main(config_path="../config", config_name="dataload")
    print(cfg.keys())
    print(cfg['dataload'])
    #print(cfg['pipeline'])
    #print(a())
sth()