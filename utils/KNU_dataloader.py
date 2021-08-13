from utils import data_io as dataio


import numpy as np 
import os 


class CoMADataset(dataio.AbstractDataset):
    """
        coma_dataset struct detail
        coma_dataset
            |- data
                |-data_(0).npy
                |   .
                |   .
                |   .
                |-data_(n).npy
            |- examples
                |- 0.obj
                |   .
                |   .
                |   .
                |- n.obj
            |- ref
                |- data.obj
            |- rig_file
                |- rig_point.txt (default)


    """
    def __init__(self, load_root_path, rig_file_name="rig_point.txt"):
        self.load_root_path = load_root_path
        self.load_train_data_path = os.path.join(self.load_root_path, "data")
        self.example_path = os.path.join(self.load_root_path,"examples")
        self.ref_file_path = os.path.join(self.load_root_path,"ref", "data.obj")
        self.rig_file_path = os.path.join(self.load_root_path,"rig_file" ,rig_file_name)
        
    @staticmethod
    def save_data(data, reference_facet, output_directory, prefix_name):
        if len(data.shape) == 2:
            data = np.expand_dims(data, 0)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for idx, v in enumerate(data):
            dataio.save_mesh(
                                os.path.join(output_directory, prefix_name+f"{idx}.obj"), 
                                v, reference_facet
                            )

        
        return True


    def prepcoess_dataset(self):
        pass

    def load_data(self):
        """

            return :
            =================
            dict()
                data_fname      : [data]'s file names.
                
                X               : [N, len(rig_data) * dims(3)]               ====> TRAINING DATASET
                Ground_truth    : train dataset [N, vertice_size, dims(3)]  ====> TRAINING DATASET

                example         : [M, vertice_size, dims(3)]
                ref             : dict type = { v : [vertice_size, dims(3)], f : [face_size, tri(3)] }
                rig_file        : list of rigdata ex) [1, 2, 3 .... k]

        """
        self.names, self.train_data_orig = dataio.load_npy_from_directory(self.load_train_data_path) # load npy dataset
        self.train_data = np.concatenate(self.train_data_orig, axis=0)
        print("shape", self.train_data.shape)
        r_v, r_f = dataio.load_mesh(self.ref_file_path) # (v, f)
        self.ref_mesh = {"v" : r_v, "f" : r_f}
        _, self.example_mesh, _ = dataio.load_mesh_from_directory(self.example_path)
        self.example_mesh = np.array(self.example_mesh)

        self.rig_data = dataio.load_rig_data(self.rig_file_path)
        self.rig_data_size = len(self.rig_data)
        
        self.X = self.train_data[:, self.rig_data, :].reshape(-1, self.rig_data_size*3) # [N, len(rigdata)*dims(3)]
        self.GT = self.train_data # ground_truth

        
        return { 
                "data_fname" : self.names, "X" : self.X, "Ground_Truth" : self.GT, 
                "examples" : self.example_mesh, "ref" : self.ref_mesh, 
                "rig_data" : self.rig_data
                }
        

    
        
        


