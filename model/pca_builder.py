import numpy as np 
from sklearn.decomposition import PCA
import functools as f

import igl 



# helper function
def mesh_flatten(x):
    """

        mesh_dataset [N, vert_nums ,3]
        we reshape matrix as bellow.

        [N_0_x  N_1_x   ... N_n_x]
        [  |      |     ...   |  ]
        [N_0_y  N_1_y   ... N_n_y]
        [  |      |     ...   |  ]
        [N_0_z  N_1_z   ... N_n_z]
        [  |      |     ...   |  ]
            
    """
    N, V, dims = x.shape

    mesh_data = x
    mesh_data = np.transpose(mesh_data, axes=[2,1,0])
    mesh_data = mesh_data.reshape(-1, N)
    return mesh_data

def mesh_unflatten(x, vertice_size):
    """

            mesh_dataset [ 3*verts_num, N ]
            we reshape matrix as bellow.
                                            [
            [N_0_x  N_1_x   ... N_n_x]        [x0 y0 z0]
            [  |      |     ...   |  ]        [   |    ]
            [N_0_y  N_1_y   ... N_n_y]   =>   [   |    ]
            [  |      |     ...   |  ]        [   |    ]
            [N_0_z  N_1_z   ... N_n_z]        [   |    ]
            [  |      |     ...   |  ]        [xn yn zn]
                                                
                                                .
                                                .
                                                .

                                              [x0 y0 z0]
                                              [   |    ]
                                              [   |    ]
                                              [   |    ]
                                              [   |    ]
                                              [xn yn zn]
                                            ]
    """
    verticeXdims, N = x.shape
    assert 3*vertice_size*N == verticeXdims*N, "wrong shape"
    x = x.reshape(3, vertice_size, N)
    x = np.transpose(x, axes=[2,1,0])

    return x



class BlendShapePcaBuilder():
    def __init__(self, **kwargs):
        """
            n_component : int
            threshold : float
        """

        if "n_component" in kwargs :
            self.n_component = kwargs.get("n_component", 3)
        




    def _create_PCA(self, mesh_dataset):
        """

            mesh_dataset [N, vert_nums ,3]
            we reshape matrix as bellow.

            [N_0_x  N_1_x   ... N_n_x]
            [|      |       ... |    ]
            [N_0_y  N_1_y   ... N_n_y]
            [|      |       ... |    ]
            [N_0_z  N_1_z   ... N_n_z]
            [|      |       ... |    ]
            


        """
        N, V, dims = mesh_dataset.shape

        assert dims == 3, "vertice dims is not 3."

        mesh_data = mesh_flatten(mesh_dataset.astype(np.float64))
        pca = None 
        if hasattr(self, "n_component"):
            pca = PCA(self.n_component)
        
        print(self.n_component)
        unflat_function = f.partial(mesh_unflatten, vertice_size=V)
        print(mesh_data.shape)
        print(mesh_data)
        # pca.fit(mesh_data)
        pca = pca.fit(mesh_data.T)

        return pca, unflat_function
