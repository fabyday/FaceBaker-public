from genericpath import exists
from utils import data_io as dataio
from utils import sort as usort
from shutil import copy2

import numpy as np 
import os 
import yaml
import igl
class KNUDataset(dataio.AbstractDataset):


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
    # CONSTANT
    __SRC = 0
    __DST = 1



    # allowed mesh file type
    __MESH_EXT = ".obj"

    # dir names
    __data_directory = ("objs", "data") # input
    __weights_directory = "weights" # ground_truth

    __ref_directory = ("reference", "ref")
    __exaples_directory = "examples" # examples.

    __rig_directory = "rig_file" 

    __rig_file_name = "rig_point.txt" 
    __ref_file_name = "generic_neutral_mesh.obj"
    __weights_names_file = "weightnames.txt" # ground truth


    # data file layout
    __INPUT = "inputs"
    __GROUND_TRUTH = "outputs" # ground_truth => weights



    __default_name = "data"

    def __init__(self, load_root_path, rig_file_name="rig_point.txt"):
        self.load_root_path = load_root_path
        self.load_train_data_path = os.path.join(self.load_root_path, "data")
        self.example_path = os.path.join(self.load_root_path,"examples")
        self.ref_file_path = os.path.join(self.load_root_path,"ref", "generic_neutral_mesh.obj")
        self.rig_file_path = os.path.join(self.load_root_path, KNUDataset.__rig_directory, rig_file_name)
        
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

    @staticmethod
    def preprocess_dataset(input_path, output_path, **kwargs):
        """
            input_path : root path of dataset directory.
            output_path : root path of output.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError("input_path is not found {}".format(input_path))
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        infos = {name : "knu"}
        with open(os.path.join(output_path, "file_info.yaml"), 'w') as f:
            yaml.dump(infos, f)
        


        # ==========================
        # process reference(JUST COPY IT TO OUTPUT DIRECTORY WITH LAYOUT.)
        # ==========================
        src_ref_parent_path = os.path.join(input_path, KNUDataset.__ref_directory[KNUDataset.__SRC])
        dst_ref_parent_path = os.path.join(output_path, KNUDataset.__ref_directory[KNUDataset.__DST])
        if not os.path.exists(dst_ref_parent_path):
            os.makedirs(dst_ref_parent_path)
        
        copy2(                    # copy file from src to dst.
                os.path.join(src_ref_parent_path, KNUDataset.__ref_file_name), 
                os.path.join(dst_ref_parent_path, KNUDataset.__ref_file_name)
                ) 
        print("reference copy...")
        


        print("rig data copy ...")
        # ==========================
        #  process rig (JUST COPY IT TO OUTPUT DIRECTORY WITH LAYOUT.)
        # ==========================
        if not exists(os.path.join(output_path, KNUDataset.__rig_directory)): #check output directory exists.
                os.makedirs(os.path.join(output_path, KNUDataset.__rig_directory))

        if os.path.exists(os.path.join(input_path, KNUDataset.__rig_directory, KNUDataset.__rig_file_name)): # check input file exitst
            
            copy2(                    # copy file from src to dst.
                        os.path.join(input_path, KNUDataset.__rig_directory, KNUDataset.__rig_file_name),
                        os.path.join(output_path, KNUDataset.__rig_directory, KNUDataset.__rig_file_name)
                        ) 
        print("rig data copy end ...")

        # ==========================
        # process examples 
        # ==========================
        print("process examples...")
        w_name_list = dataio.load_text_data(os.path.join(input_path, KNUDataset.__weights_names_file))
        npy_examples = []
        for wname in w_name_list:
            v, _ = igl.read_triangle_mesh(os.path.join(input_path, KNUDataset.__exaples_directory, wname+KNUDataset.__MESH_EXT))
            npy_examples.append(v)
        
        npy_examples = np.array(npy_examples) # [ N, Verts_num, dims(3) ] 
        if not os.path.exists(os.path.join(output_path, KNUDataset.__exaples_directory)):
            os.makedirs(os.path.join(output_path, KNUDataset.__exaples_directory))
        dataio.save_npy(os.path.join(output_path, KNUDataset.__exaples_directory, KNUDataset.__default_name+".npy"), npy_examples)
        print("process examples end...")


        # ==========================
        # process objs and weights 
        # ==========================
        print("process data ...")
        i_mesh_names, i_vertice, _ = dataio.load_mesh_from_directory(
                                                                    dname=os.path.join(input_path, KNUDataset.__data_directory[KNUDataset.__SRC]),
                                                                    comp = usort.alphanum_comp
                                                                    )
        i_vertice = np.array(i_vertice) # N, verts_size, dims(3)
        print("process data end ...")

        # process weights
        print("process weights ...")
        load_function = lambda fname : (fname, dataio.load_weights_data(fname))
        def save_function(tmp_loc, *args):
            r_obj = range(len(args))
            if len(tmp_loc) == 0 : 
                for _ in r_obj:
                    tmp_loc.append([])
            for idx, d in enumerate(args):
                tmp_loc[idx].append(d)
            
        dname = os.path.join(input_path, KNUDataset.__weights_directory)
        w_mesh_names, weights = dataio.load_file_from_directory(
                                                                dname=dname,
                                                                ext_list =[".txt"],
                                                                load_function=load_function,
                                                                save_function=save_function,
                                                                comp=usort.alphanum_comp
                                                                )
        weights = np.array(weights) # N, 1, weights_num
        print("process weights end ...")

        # check i_mesh and weights name 
        # assert len(weights)==len(i_vertice), "shape is diff."
        for i, j  in zip(i_mesh_names, w_mesh_names):
            
            assert os.path.basename(i).split(".")[0] == os.path.basename(j).split(".")[0] , "data is not same ...{}, {}".format(os.path.basename(i).split(".")[0], os.path.basename(j).split(".")[0])
        
        print("save data ...")

        if not os.path.exists(os.path.join(output_path, KNUDataset.__data_directory[KNUDataset.__DST])):
            os.makedirs(os.path.join(output_path, KNUDataset.__data_directory[KNUDataset.__DST]))
        save_dataset = {
                        KNUDataset.__INPUT : i_vertice,
                        KNUDataset.__GROUND_TRUTH : weights
                        }
        dataio.save_npz(
                        os.path.join(output_path, KNUDataset.__data_directory[KNUDataset.__DST], KNUDataset.__default_name)+".npz", 
                        **save_dataset
                        # inputs=i_vertice, 
                        # outputs=weights
                        )

    
        print("data preprocess end ...")

    def load_data(self):
        """

            return :
            =================
            dict()
                data_fname          : [data]'s file names.

                X                   : [N, len(rig_data) * dims(3)]               ====> TRAINING DATASET
                Ground_Truth        : train dataset [N, vertice_size, dims(3)]  ====> TRAINING DATASET
                Ground_Truth_Weight : [N, 1, examples_sizes ] ====> TRAINING DATASET (Mesh IK weights or Direct manipulation)
                example             : [M, vertice_size, dims(3)]
                ref                 : dict type = { v : [vertice_size, dims(3)], f : [face_size, tri(3)] }
                rig_file            : list of rigdata ex) [1, 2, 3 .... k]

        """
        self.names, self.train_data_orig = dataio.load_npy_from_directory(self.load_train_data_path) 
        load_function = lambda fname : (fname, dataio.load_npz(fname))
        def save_function(tmp_loc, *args):
            r_obj = range(len(args))
            if len(tmp_loc) == 0 : 
                for _ in r_obj:
                    tmp_loc.append([])
            for idx, d in enumerate(args):
                tmp_loc[idx].append(d)

        self.names, self.train_data_orig = dataio.load_file_from_directory( # load npz dataset ['inputs' : list [ndarray], 'outputs : list[ndarray] ]
                                                                            dname = self.load_train_data_path,
                                                                            ext_list=[".npz"],
                                                                            load_function = load_function,
                                                                            save_function=save_function,
                                                                            comp=usort.alphanum_comp
                                                                        )
        self.train_data_verts = [dataset[KNUDataset.__INPUT] for dataset in self.train_data_orig]
        self.ground_truth_weights = [dataset[KNUDataset.__GROUND_TRUTH] for dataset in self.train_data_orig]


        self.train_data = np.concatenate(self.train_data_verts, axis=0)
        self.ground_truth_weights = np.concatenate(self.ground_truth_weights, axis=0)

        r_v, r_f = dataio.load_mesh(self.ref_file_path) # (v, f)
        self.ref_mesh = {"v" : r_v, "f" : r_f}

        
        self.example_mesh = dataio.load_npy(os.path.join(self.example_path, KNUDataset.__default_name+".npy"))
        

        self.rig_data = dataio.load_rig_data(self.rig_file_path)
        self.rig_data_size = len(self.rig_data)
        
        self.X = self.train_data[:, self.rig_data, :].reshape(-1, self.rig_data_size*3) # [N, len(rigdata)*dims(3)]
        self.GT = self.train_data # ground_truth

        
        return { 
                "data_fname" : self.names, "X" : self.X, "Ground_Truth" : self.GT, 
                "Ground_Truth_Weight" : self.ground_truth_weights,
                "examples" : self.example_mesh, "ref" : self.ref_mesh, 
                "rig_data" : self.rig_data
                }
        

    
        
        


