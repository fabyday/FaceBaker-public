import tensorflow as tf
import tensorflow.keras as K 
from model.pca_builder import BlendShapePcaBuilder
# see this paper. https://graphics.pixar.com/library/FaceBaker/paper.pdf


class TF_PCA(K.layers.Layer):
    """
        PCA layer is fixed(it doesn't allow to train this matrix). it was precomputed by scikit-learn module. 
        see also, BlendShapePcaBuilder in pca.py
        
    """
    def __init__(self, components, mean, vertice_size, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.components_ = tf.constant(components, dtype=tf.float32, shape=components.shape, name="pca_components")
        self.mean_ = mean = tf.constant(mean, dtype=tf.float32, shape=mean.shape, name="pca_means")


        self.output_reshape = kwargs.get("output_reshape", False)

        self.vertice_size = vertice_size
    

    def call(self, x):
        """
            this is PCA-inverse_transform(x) in scikit-learn
            x      : [ batch_size, components_size]
            ==================================
            output : [ batch_size, vertice_num, dims(3) ]
        """
        reval = tf.matmul(x, self.components_) + self.mean_
        if self.output_reshape:
            reval = tf.reshape(reval, (3, self.vertice_size, -1))
            reval = tf.transpose(reval, perm=[2,1,0])
        return reval



@tf.function
def pca_inverse(x):
    np.dot(X, self.components_) + self.mean_

class ModelBuilder:
    def __init__(self,
                 mesh_dataset,
                 checkpoint_root,
                 n_component = 50,
                 dense_layers_width=256,
                 num_dense_layers=8,
                 leaky_relu_alpha=0.3,
                 loss = 'mse',
                 optimizer='adam',
                 use_numeric = False
                 ):
        """
            mesh_dataset : precomputing inverse matrix using PCA
            checkpoint_root : checkpoint root dir
            n_component : number of PCA components (num_rig_control_variables)
            dense_layer_width : output of each layers. default = 256,  according to the paper.
            num_dense_layer : number of dense layers. default = 8, according to the paper.
            leaky_relu_alpha : Since it's not clear from the paper, the default is 0.3.
            
            optimizer : default Adam Optimizer
            loss      : default Mean-Squared-Error (L2)

            use_numeric : if True, last layer use PCA matrix directly





        """
        self.dense_layers_width = dense_layers_width
        self.num_dense_layers = num_dense_layers
        self.leaky_relu_alpha = leaky_relu_alpha
        
        self.model = None
        self.checkpoint_root = checkpoint_root


        self.loss = loss 
        self.optimizer = optimizer

        self.use_numeric = use_numeric

        _, self.input_verts_size, _ = mesh_dataset.shape


        self.n_component = n_component
        self.pca, self.unflat_function = BlendShapePcaBuilder(n_component=n_component)._create_PCA(mesh_dataset)
        print(self.pca.explained_variance_ratio_)
        print("sin val : \n",self.pca.singular_values_)
        
        self.component_ = self.pca.components_
        print(self.component_.shape)
        self.mean_ = self.pca.mean_


    @staticmethod
    def dense_layer_name(layer_num):
        return f'Dense_{layer_num}'

    
    def _make_or_restore_model(self, check_pointroot, num_rig_control_variables):
        # Either restore the latest model, or create a fresh one
        # if there is no checkpoint available.
        import os 
        checkpoints = [check_pointroot + "/" + name for name in os.listdir(check_pointroot)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            return K.models.load_model(latest_checkpoint), int(latest_checkpoint[latest_checkpoint.find("cp-")+3 : latest_checkpoint.find(".ckpt")])
        print("Creating a new model")
        return self._get_compiled_model(num_rig_control_variables), 0
    


    def _get_compiled_model(self, num_rig_control_variables):
        dims = 3
        input_layer = K.Input(shape=(num_rig_control_variables * dims ,), name="Input_Rig_Control_Variables")

        # Add the first Dense layer
        dense_layer = K.layers.Dense(self.dense_layers_width, name=self.dense_layer_name(1))(input_layer)

        # We don't need skip connections after the first Dense layer
        skip_layer = None

        # Add the rest 7 layer blocks
        for layer_num in range(2, self.num_dense_layers + 1):
            dense_layer, skip_layer = self._add_dense_layer_with_skip_connections(layer_num, dense_layer, skip_layer)

        # Add missing Leaky ReLU
        leaky_relu = K.layers.LeakyReLU(alpha=self.leaky_relu_alpha)(skip_layer)

        # Add PCA Dense layer
        pca_layer = K.layers.Dense(self.n_component, name="PCA")(leaky_relu) # 8-th layer
        #it is fixed.
        output_layer1 = TF_PCA(components=self.component_ , mean=self.pca.mean_, 
                            name="output_PCA", 
                            vertice_size = self.input_verts_size)




        ################# __LAST_LAYER__
        if self.use_numeric : 
            output_layer = TF_PCA(components=self.component_ , mean=self.pca.mean_, 
                             name="output_PCA", 
                             vertice_size = self.input_verts_size)(pca_layer)
        else : 
            output_layer = K.layers.Dense(self.input_verts_size * 3, name="Output_Mesh_Coordinates")(pca_layer) #9-th layer
        # output_layer = K.layers.Dense(self.input_verts_size * 3, name="Output_Mesh_Coordinates")(pca_layer) #9-th layer
        # output_layer = K.layers.Reshape((self.input_verts_size ,3))(output_layer)
        # Create the model
        

        self.model = K.Model(inputs=input_layer, outputs=output_layer, name="FaceBaker")


        self.model.compile(optimizer = self.optimizer, loss=self.loss)

        self.model.summary()
        print("model build complete...")

        return self.model

    def create_model(self, num_rig_control_variables = 100):
        print("model build ...")
        # Add the flat input layer
        model, self.current_epochs = self._make_or_restore_model(self.checkpoint_root, num_rig_control_variables)
        
        return model, self.current_epochs , self.unflat_function
        
        

    def _add_dense_layer_with_skip_connections(
            self,
            layer_num,
            dense_layer_prev,
            skip_layer_prev):

        """
        Add a Dense - Leaky ReLU - Skip Connection block as in Figure 2
        """
        
        # After the first Dense layer we don't have a subsequent Add layer, so use that Dense layer instead
        if skip_layer_prev is None:
            skip_layer_prev = dense_layer_prev

        leaky_relu = K.layers.LeakyReLU(alpha=self.leaky_relu_alpha)(skip_layer_prev)
        dense_layer_new = K.layers.Dense(self.dense_layers_width, name=self.dense_layer_name(layer_num))(leaky_relu)
        skip_layer_new = K.layers.Add()([dense_layer_prev, dense_layer_new])

        return dense_layer_new, skip_layer_new

    def plot_model_to_file(self, file_path):
        K.utils.plot_model(self.model, file_path, show_shapes=True)




if __name__ == "__main__":
    import igl 
    import numpy as np 
    v1, F =  igl.read_triangle_mesh("pca_test/0.obj")
    v2, _  = igl.read_triangle_mesh("pca_test/1.obj")
    v3, _  = igl.read_triangle_mesh("pca_test/2.obj")
    v4, _  = igl.read_triangle_mesh("pca_test/3.obj")
    v = np.stack([v1,v2,v3,v4], axis=0)
    print(v.shape)
    model = ModelBuilder(v,4).create_model(num_rig_control_variables=100)
    model.compile()
    model.fit()