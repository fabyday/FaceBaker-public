from locale import Error
from re import search
import numpy as np 
import os , glob

import igl
import functools
from inspect import signature

from numpy.lib.npyio import load
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray

from utils import file_io_helper as fio


__mesh_data_type = [".obj", ".ply"]
__numpy_npy_type = ".npy"
__numpy_npz_type = ".npz"

__comment_str = "#"


class AbstractDataset(object):
    def __init__(self, load_path):
        pass

    def load_data(self):
        raise NotImplementedError()
    
    def save_data(self):
        raise NotImplementedError()


def load_npy(fname : str):
    with open(fname, mode="rb") as f:
        return np.load(f, allow_pickle=True)
    
def load_rig_data(fname : str):
    buff_size = 256
    with open(fname, "r") as f:
        num_str = f.readline(buff_size)
        num = num_str.split(__comment_str)[0]
        size = int(num)
        reval = []
        for _ in range(size):
            reval.append(int(f.readline(buff_size)))
            
    return reval    


def save_mesh(fname : str, v : ndarray, f : ndarray):
    igl.write_triangle_mesh(fname, v, f)
    return True



def load_mesh(fname : str ):
    """
        wrapper function of igl.read_triangle_mesh()
        ============================================
        return vertice, facet
    """
    return igl.read_triangle_mesh(fname)


def load_file_from_directory(dname : str, 
                             ext_list : list, 
                             load_function, 
                             save_function,
                             comp=None, 
                             recursive=True, 
                             glob_eval_lazy=True
                             ):
    """
        dname : directory name
        ext_list : extension list 
        load_function : data load function. it return loaded data. 
                        its output params are used as an arguments to the next function(save_function).
        save_function : save data after load data. 
                        function params is save_function(tmp_reval, lf_re1, lf_re2.. lf_re_n)
        comp : custom sort function. if none, it doesn't concearn sort.
        recursive : recursively find files. 
        glob_eval_lazy : if True, internally glob.iglob is used. or False, glob.glob is used.
        ==================================================================
        return 
            mesh_names     : [ N ] list 
            mesh_vertice   : [ N, vertice_size, dims(3) ] ndarray
            mesh_face      : [ N, face_size, tri(3) ]  ndarray
    """

    if not os.path.exists(dname) or not os.path.isdir(dname):
        raise FileNotFoundError("directory is not exists")
    
    if not hasattr(load_function, "__call__"): # check is it CALLABLE?(Function, is it?)
        if not len(signature(load_function).parameters) == 1 : # check! does it have 2 params? comp(a, b) -> int
            raise TypeError("argument is needed 1, but {}".format(signature(load_function).parameters))
        raise TypeError("it is not Callable Object or function")
    
            
    

    search_pattern = [os.path.join(dname, "**", ext) for ext in ext_list ]
    
    
    # __MAKE_NAME_LIST__
    file_names = None
    if glob_eval_lazy :
        file_names = fio.iglob_multiple_file_type(*search_pattern, recursive=recursive)
    else :
        file_names = fio.glob_multiple_file_type(*search_pattern, recursive=recursive)
    
    
    # __SORT_NAME__
    if hasattr(comp, "__call__"): # check is it CALLABLE?(Function, is it?)
        if len(signature(comp).parameters) == 2 : # check! does it have 2 params? comp(a, b) -> int
            file_names = sorted(file_names, key=functools.cmp_to_key(comp)) # if yes! data list is sorted by comp function
    

    # __LOAD_MESH_DATA__
    reval = None
    for m_name in file_names:
        s = load_function(m_name)
        save_function(reval, s)

    return file_names, reval


def load_npy_from_directory(dname : str, comp=None, recursive=True, glob_eval_lazy=True):
    """
        dname : directory name
        comp : custom sort function. if none, it doesn't concearn sort.
        recursive : recursively find files. 
        glob_eval_lazy : if True, internally glob.iglob is used. or False, glob.glob is used.
        ==================================================================
        return 
            mesh_names     : [ N ] list 
            mesh_vertice   : [ N, vertice_size, dims(3) ] ndarray
            mesh_face      : [ N, face_size, tri(3) ]  ndarray
    """

    if not os.path.exists(dname) or not os.path.isdir(dname):
        raise FileNotFoundError("directory is not exists")
    

    search_pattern = [os.path.join(dname, "**", "**"+__numpy_npy_type)]
    
    # __MAKE_NAME_LIST__
    npy_files = None
    if glob_eval_lazy :
        npy_files = fio.iglob_multiple_file_types(*search_pattern, recursive=recursive)
    else :
        npy_files = fio.glob_multiple_file_types(*search_pattern, recursive=recursive)
    
    
    # __SORT_NAME__
    if hasattr(comp, "__call__"): # check is it CALLABLE?(Function, is it?)
        if len(signature(comp).parameters) == 2 : # check! does it have 2 params? comp(a, b) -> int
            npy_files = sorted(npy_files, key=functools.cmp_to_key(comp)) # if yes! data list is sorted by comp function
    

    # __LOAD_MESH_DATA__
    npy_data = [] # vertex_list
    for npy_name in npy_files:
        npy = load_npy(npy_name)
        npy_data.append(npy)
        
    

    return npy_files, npy_data


def load_mesh_from_directory(dname : str, comp=None, recursive=True, glob_eval_lazy=True):
    """
        dname : directory name
        comp : custom sort function. if none, it doesn't concearn sort.
        recursive : recursively find files. 
        glob_eval_lazy : if True, internally glob.iglob is used. or False, glob.glob is used.
        ==================================================================
        return 
            mesh_names     : [ N ] list 
            mesh_vertice   : [ N, vertice_size, dims(3) ] ndarray
            mesh_face      : [ N, face_size, tri(3) ]  ndarray
    """

    if not os.path.exists(dname) or not os.path.isdir(dname):
        raise FileNotFoundError("directory is not exists")

    search_pattern = [os.path.join(dname, "**", "**"+ext) for ext in __mesh_data_type ]
    
    # __MAKE_NAME_LIST__
    mesh_names = None
    if glob_eval_lazy :
        mesh_names = fio.iglob_multiple_file_types(*search_pattern, recursive=recursive)
    else :
        mesh_names = fio.glob_multiple_file_types(*search_pattern, recursive=recursive)
    
    
    # __SORT_NAME__
    if hasattr(comp, "__call__"): # check is it CALLABLE?(Function, is it?)
        if len(signature(comp).parameters) == 2 : # check! does it have 2 params? comp(a, b) -> int
            mesh_names = sorted(mesh_names, key=functools.cmp_to_key(comp)) # if yes! data list is sorted by comp function
    

    # __LOAD_MESH_DATA__
    mesh_v_list = [] # vertex_list
    mesh_f_list = [] # face list
    print(search_pattern)
    for m_name in mesh_names:
        v,f = load_mesh(m_name)
        mesh_v_list.append(v)
        mesh_f_list.append(f)

    return mesh_names, mesh_v_list, mesh_f_list

