# MODEL_CONFIG_PATH = '../fragnet_edge/exps/ft/esol/fragnet_hpdl_exp1s_pt4_30'
# MODEL_CONFIG = f'{MODEL_CONFIG_PATH}/config_exp100.yaml'
# MODEL_PATH = f'{MODEL_CONFIG_PATH}/ft_100.pt'


MODEL_CONFIG_PATH = '../fragnet_edge/exps/ft/pnnl_set2/'
MODEL_CONFIG = f'{MODEL_CONFIG_PATH}/exp1s_h4pt4.yaml'
MODEL_PATH = f'{MODEL_CONFIG_PATH}/h4/ft.pt'


PROP_LIST=[
    "Solubility",
    "Solubility (tuned)", 
    # "Solubility (Multi-Frag)",
    "Solubility (Multi-Frag, tuned)", 
    "Lipophilicity", 
    "Energy", 
    "DRP"]

PROP_NAME_LIST=[
    "In logS units",
    "In logS units", 
    # "In logS units",
    "In logS units",  
    "Lipophilicity", 
    "Energy", 
    "Drug Response Prediction"]



