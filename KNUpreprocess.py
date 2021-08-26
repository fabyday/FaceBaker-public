import utils as mesh
if __name__ == "__main__":
    import argparse 
    import os
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", "--input", type=str, default="data")

    parser.add_argument("-mn", "--data_model_name", type=str, default="KNU_face_dataset")
    parser.add_argument("-o", "--output", type=str, default="train_dataset")

    
    args = parser.parse_args()
    mesh.KNUDataset.preprocess_dataset(os.path.join(args.input, args.data_model_name),
                    os.path.join(args.output, args.data_model_name)
                    )

    


    
