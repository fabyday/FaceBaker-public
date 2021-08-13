import tensorflow as tf
import tensorflow.keras as K 
from pca import BlendShapePcaBuilder
# see this paper. https://graphics.pixar.com/library/FaceBaker/paper.pdf


class TF_PCA(K.layers.Layer):
    """
        PCA layer is fixed(it doesn't allow to train this matrix). it was precomputed by scikit-learn module. 
        see also, BlendShapePcaBuilder in pca.py
        
    """
    def __init__(self, components, mean, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.components_ = tf.constant(components, dtype=tf.float32, shape=components.shape, name="pca_components")
        self.mean_ = mean = tf.constant(mean, dtype=tf.float32, shape=mean.shape, name="pca_means")

    

    def call(self, x):
        """
            this is PCA-inverse_transform(x) in scikit-learn
            x      : [ batch_size, components_size]
            ==================================
            output : [ batch_size, vertice_num, dims(3) ]
        """
        return tf.matmul(x, self.components_) + self.mean_



@tf.function
def pca_inverse(x):
    np.dot(X, self.components_) + self.mean_

class ModelBuilder:
    def __init__(self,
                 mesh_dataset,
                 n_component = 50,
                 dense_layers_width=256,
                 num_dense_layers=8,
                 leaky_relu_alpha=0.3
                 ):
        """
            mesh_dataset : precomputing inverse matrix using PCA
            n_component : number of PCA components 
            dense_layer_width : output of each layers. default = 256,  according to the paper.
            num_dense_layer : number of dense layers. default = 8, according to the paper.
            leaky_relu_alpha : Since it's not clear from the paper, the default is 0.3.
        """
        self.dense_layers_width = dense_layers_width
        self.num_dense_layers = num_dense_layers
        self.leaky_relu_alpha = leaky_relu_alpha
        
        self.model = None


        self.n_component = n_component
        self.pca, self.unflat_function = BlendShapePcaBuilder(n_component=n_component)._create_PCA(mesh_dataset)

        self.component_ = self.pca.components_
        print(self.component_.shape)
        self.mean_ = self.pca.mean_


    @staticmethod
    def dense_layer_name(layer_num):
        return f'Dense_{layer_num}'

    def create_model(
            self,
            num_rig_control_variables=100
            ):
        print("model build ...")
        # Add the flat input layer
        input_layer = K.Input(shape=(num_rig_control_variables,), name="Input_Rig_Control_Variables")

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
        output_layer = TF_PCA(components=self.component_ , mean=self.pca.mean_, name="output_PCA")(pca_layer)

        # Add output Dense layer
        # output_layer = keras.layers.Dense(num_output_mesh_vertices * 3, name="Output_Mesh_Coordinates")(pca_layer) #9-th layer

        # Create the model
        
        self.model = K.Model(inputs=input_layer, outputs=output_layer, name="FaceBaker")
        print("model build complete...")
        self.model.summary()
        return self.model

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



