    
    
    
import argparse

from tensorflow.keras import callbacks
    
import model

import utils
import tf_utils
import tensorflow as tf
import yaml

# 

def main(**kwargs):

    import numpy as np 
    import utils
    import os 

    
    
    checkpoint_root = os.path.join(kwargs['checkpoint_dir'], kwargs['name'])

    print(checkpoint_root)
    checkpoint_callback = tf_utils.TFcheckpoint_callback_builder(checkpoint_root)
    
    tmp_mode = kwargs['mode']
    #configure serialization
    if os.path.exists(os.path.join(checkpoint_root, "model.conf")):
        with open(os.path.join(checkpoint_root, "model.conf")) as f:
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
    else : # write 
        with open(os.path.join(checkpoint_root, "model.conf"), 'w') as f:
            yaml.dump(kwargs, f)
    kwargs['mode'] = tmp_mode


    dataLoader = utils.Dataset_Factory.Build_from_options(input=kwargs['input'], rig_point_path=kwargs['rig_point_path'])

    # dataset = utils.CoMADataset(kwargs['input'], kwargs['rig_point_path']).load_data()
    dataset = dataLoader.load_data()



    n_component = len(dataset['examples'])
    n_component = 8
    print(n_component)


      
    facebaker_model, current_epochs, unflat_f = model.ModelBuilder( dataset['examples'], 
                                                                    checkpoint_root = checkpoint_root,
                                                                    n_component = n_component ,
                                                                    optimizer = kwargs["optimizer"], 
                                                                    loss=kwargs["loss"],
                                                                    use_numeric=kwargs['use_numeric']
                                                                    ).create_model(num_rig_control_variables=len(dataset['rig_data']))


    if kwargs['mode'] == "train":
        print("data size ", len(dataset['X']))
        facebaker_model.fit(  
                            x=dataset['X'], 
                            batch_size = kwargs['batch_size'],
                            y=model.mesh_flatten(dataset["Ground_Truth"]).T, epochs=kwargs['epochs'], callbacks=[checkpoint_callback], 
                            # y=dataset["Ground_Truth"], epochs=kwargs['epochs'], callbacks=[checkpoint_callback], 
                            validation_split=kwargs['validation_split'], initial_epoch = current_epochs
                            )
    elif kwargs['mode'] == 'test':
        
        
        result = facebaker_model.predict(
                                        x = dataset['X'],
                                        batch_size = kwargs['batch_size']
                                        )
        print(result.shape)
        dataLoader.save_data(
                                    dataset["Ground_Truth"], reference_facet=dataset['ref']['f'], 
                                    output_directory=os.path.join(kwargs['output'], kwargs['name']),
                                    prefix_name="input_"
                                    )
        dataLoader.save_data(
                                    unflat_f(result.T), reference_facet=dataset['ref']['f'], 
                                    output_directory=os.path.join(kwargs['output'], kwargs['name']),
                                    prefix_name="pred_"
                                    )
        # utils.CoMADataset.save_data(
        #                             dataset["Ground_Truth"], reference_facet=dataset['ref']['f'], 
        #                             output_directory=os.path.join(kwargs['output'], kwargs['name']),
        #                             prefix_name="input_"
        #                             )
        # utils.CoMADataset.save_data(
        #                             unflat_f(result.T), reference_facet=dataset['ref']['f'], 
        #                             output_directory=os.path.join(kwargs['output'], kwargs['name']),
        #                             prefix_name="pred_"
        #                             )













    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Baker')
    parser.add_argument('-i', '--input', type=str, default="./train_dataset/coma_dataset") # input dataset
    parser.add_argument('-n', '--name', type=str, default="super_duper") # model name
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test']) 
    parser.add_argument('-o', '--output', type=str, default="./result") # output
    parser.add_argument('-c', '--checkpoint_dir', type=str, default="./checkpoints") # output
    parser.add_argument('-opt', '--optimizer', type=str, choices=["Adam"], default="Adam") # output
    parser.add_argument("-l", "--loss", type=str, default="mse")
    parser.add_argument("-r", "--rig_point_path", type=str, default="rig_point.txt")
    parser.add_argument("-b", "--batch_size", type=int, default=5)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument("-v", "--validation_split", type=float, default=0.2 )
    
    
    
    # for testing. default := False. 
    # if it is true, last layer use PCA numerical weights from example meshes. 
    # if it is False, last layer just Dense Layer.
    parser.add_argument("--use_numeric", action='store_true' , default=False) 


    parser.add_argument
    


    kwargs = vars(parser.parse_args())
    main(**kwargs)