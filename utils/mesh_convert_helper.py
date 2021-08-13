import igl
import numpy  as np


import glob, os


def mesh2numpy(input_path, output_path, search_ext, fname_prefix = "data", size=1000, discard_res=False):
    """
        input_path : data input path(root dir)
        output_path : path where data is saved.
        search_ext : what you want to find file extension.
        fname_prefix : npy file name prefix. default is data_"i-th".npy
        size : packing size of dataset. default is 1000.
        discard_res : Discard residual data sets smaller than the specified size.   
        =========================================================================
        Return 
            bool :  
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError("directory is not exists.")
    print(output_path)
    if not os.path.exists(output_path)  :
        os.makedirs(os.path.abspath(output_path))
        


    file_list = glob.glob(os.path.join(input_path,"**", f"**.{search_ext}"), recursive=True)
    # preprocess
    data = []
    for f_name in file_list :         
        v, _ = igl.read_triangle_mesh(f_name)
        data.append(v)

    # data = np.array(data)
    # post process, divide by size, and save it.
    start = 0
    r_idx = 0
    for idx, chunk_end in enumerate(range(size, len(data) + 1, size)):
        print(chunk_end)
        with open(os.path.join(output_path, f"{fname_prefix}_{idx}.npy"), "wb") as f:
            np.save(f, np.array(data[start:chunk_end]))
        start = chunk_end
        r_idx = idx


    # if residual exists and discard_res flag is False, save residual.
    if not discard_res and start != len(data):
        print("saved")
        with open(os.path.join(output_path, f"{fname_prefix}_{r_idx+1}.npy"), "wb") as f:
            np.save(f, np.array(data[start:]))


    return True

