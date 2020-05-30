import argparse
import train_helper

parser = argparse.ArgumentParser()

parser.add_argument('data_dir', 
                    action="store",
                    type=str,
                    help='Data directory')

parser.add_argument('--save_dir', 
                    action="store", 
                    dest="save_dir",
                    type=str, 
                    default="model_bash.pt", 
                    help='Save directory')

parser.add_argument('--arch', 
                    action="store", 
                    dest="arch",
                    type=str, 
                    default="resnet50", 
                    help='Model Architecture')

parser.add_argument('--learning_rate', 
                    action="store", 
                    dest="learning_rate",
                    type=float, 
                    default=0.0005, 
                    help='learning_rate')

parser.add_argument('--hidden_units', 
                    action="store", 
                    dest="hidden_units",
                    type=int, 
                    default=512, 
                    help='Hidden units')

parser.add_argument('--epochs', 
                    action="store", 
                    dest="epochs",
                    type=int, 
                    default=20, 
                    help='Epochs')

parser.add_argument('--gpu', 
                    action="store_true", 
                    dest="gpu", 
                    default=True, 
                    help='GPU enabled')

#print(parser.parse_args())




def main(args):
    print(type(args))
    print(vars(args))
    train_helper.preprocess_and_train(vars(args))
    

if __name__ == "__main__":
    main(parser.parse_args())