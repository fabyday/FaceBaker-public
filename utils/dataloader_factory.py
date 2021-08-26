from utils import CoMA_dataloader as CoMADataset
from utils import KNU_dataloader as KNUDataset

import yaml
import os 


class Abstract_Factory:

    _KNU_DATASET_CANDIDATE = "KNU"
    _CoMA_DATASET_CANDIDATE = "COMA"


    
    _DEFAULT_CoMA_PATH = "train_dataset/KNU_face_dataset" 
    _DEFAULT_KNU_PATH = "train_dataset/coma_dataset"
    _META_DATA_FILE_NAME = "file_info.yaml"
    @classmethod
    def Build_from_options(cls, **kwargs):
        raise NotImplementedError





class Dataset_Factory(Abstract_Factory):
    """
        
    """
    @classmethod
    def Build_from_options(cls, **kwargs):
        """
            input : 
            rig_point_path : default rig_point.txt

        """

        input_path = kwargs.get("input", cls._DEFAULT_CoMA_PATH)
        meta_data = {}
        with open(os.path.join(input_path, cls._META_DATA_FILE_NAME), "r") as f:
            meta_data = yaml.load(f, Loader=yaml.FullLoader)
        build_type = meta_data.get("name", "COMA").upper()
        
        
        
        
        rig_file_name = kwargs.get('rig_point_path', "rig_point.txt")
        if build_type == cls._KNU_DATASET_CANDIDATE :
            return KNUDataset.KNUDataset(load_root_path = input_path, rig_file_name=rig_file_name)
        elif build_type == cls._CoMA_DATASET_CANDIDATE:
            return CoMADataset.CoMADataset(load_root_path = input_path, rig_file_name = rig_file_name)
        

        







