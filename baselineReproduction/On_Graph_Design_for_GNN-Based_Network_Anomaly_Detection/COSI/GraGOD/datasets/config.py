from dataclasses import dataclass
from typing import Dict, Type

from gragod import Datasets


@dataclass
class Paths:
    base_path: str


@dataclass
class SWATPaths(Paths):
    base_path: str = "/home/killian/Documents/Data/Swat"
    name_train: str = "SWaT_data_train.csv"
    name_val: str = "SWaT_data_val.csv"
    name_test: str = "SWaT_data_test.csv"
    edge_index_path: str = "datasets_files/swat/edge_index.pt"


@dataclass
class TELCOPaths(Paths):
    base_path: str = "datasets_files/telco_v1"

@dataclass
class CISCOPaths(Paths):
    base_path: str = "/home/killian/Documents/Data/cisco"
    name_train: str = "preprocessed_final_short_cisco_train.csv"
    name_val: str = "preprocessed_new_short_cisco_val.csv"
    name_test: str = "preprocessed_final_short_cisco_test.csv"
    edge_index_path: str = "/final/cosi5_l4_th6_25_short.csv"


@dataclass
class DatasetConfig:
    normalize: bool
    paths: Type[Paths]


@dataclass
class SWATConfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = SWATPaths


@dataclass
class TELCOConfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = TELCOPaths

@dataclass
class CISCOnfig(DatasetConfig):
    normalize: bool = True
    paths: Type[Paths] = CISCOPaths



def get_dataset_config(dataset: Datasets) -> DatasetConfig:
    DATASET_CONFIGS: Dict[Datasets, DatasetConfig] = {
        Datasets.SWAT: SWATConfig(),
        Datasets.TELCO: TELCOConfig(),
        Datasets.CISCO: CISCOnfig(),
    }
    return DATASET_CONFIGS[dataset]
