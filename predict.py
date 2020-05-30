import argparse
import predict_helper

parser = argparse.ArgumentParser()

parser.add_argument('path_to_image', 
                    action="store",
                    type=str,
                    help='Path to Image')

parser.add_argument('checkpoint', 
                    action="store",
                    type=str,
                    help='Checkpoint')

parser.add_argument('--arch', 
                    action="store", 
                    dest="arch",
                    type=str, 
                    default="vgg13", 
                    help='Model Architecture')

parser.add_argument('--top_k', 
                    action="store",
                    dest="top_k",
                    type=int,
                    default = 1,
                    help='top_k')

parser.add_argument('--category_names', 
                    action="store",
                    dest="category_names",
                    type=str,
                    default = 'cat_to_name.json',
                    help='Category Names')

parser.add_argument('--gpu', 
                    action="store_true", 
                    dest="gpu", 
                    default=True, 
                    help='GPU enabled')



def main(args):
    print(type(args))
    print(vars(args))
    predict_helper.load_model_and_predict(vars(args))
    

if __name__ == "__main__":
    main(parser.parse_args())