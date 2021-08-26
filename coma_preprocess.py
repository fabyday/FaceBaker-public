import utils as mesh
if __name__ == "__main__":
    import argparse 
    import os
    parser = argparse.ArgumentParser("")
    parser.add_argument("-input", "--input", type=str, default="data")

    parser.add_argument("-mn", "--data_model_name", type=str, default="coma_dataset")
    parser.add_argument("-o", "--output", type=str, default="train_dataset")
    parser.add_argument("-e", "--ext", type=str, choices=["obj","ply"], default="obj")
    parser.add_argument("-n", "--filename", type=str, default="data")
    parser.add_argument("-s", "--size", type=int, default=1000)
    parser.add_argument("-d", "--discard_res", type=bool, default=False)
    
    args = parser.parse_args()
    mesh.CoMADataset.preprocess_dataset(args.input, 
                                        os.path.join(args.output, args.data_model_name), 
                                        ext=args.ext, filename=args.filename, size=args.size, discard_res=args.discard_res
                                        )
    

    


    
