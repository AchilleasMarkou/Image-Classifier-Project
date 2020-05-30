import torch
import torchvision
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F
import json

models_dict = { 'resnet18' : models.resnet18(pretrained=True),
                'alexnet' : models.alexnet(pretrained=True),
                'squeezenet' : models.squeezenet1_0(pretrained=True),
                'vgg16' : models.vgg16(pretrained=True),
               'vgg13' : models.vgg13(pretrained=True),
               'resnet50' : models.resnet50(pretrained=True)}

def load_model_and_predict(args_dict):
    cat_to_name = load_category_names(args_dict['category_names'])
    model = load_model(args_dict['checkpoint'], args_dict['arch'], args_dict['gpu'])
    predict(args_dict['path_to_image'], model, args_dict['top_k'], cat_to_name, args_dict['gpu'])     

def load_category_names(category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name

def load_model(load_path, arch, gpu):
    # load the model that provided by the user
    model = models_dict[arch]
    
    if gpu == True:
        model = model.cuda()
    
    if arch == "vgg13" or arch == "vgg16":        
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 102)
    else:        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 102)
    
    model.load_state_dict(torch.load(load_path))
    model.class_to_idx = torch.load('class_to_idx.pt')
    
    #return the model
    return model

def process_image(image, gpu):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    use_cuda = gpu
    # preprocessing
    img = Image.open(image)
    
    newsize = (224, 224)
    img = img.resize(newsize)
    #img = img.crop((224, 224))
    np_image = np.array(img)
    
    np_image = np_image/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    
    np_image = np_image.transpose((2, 0, 1))
    
    image_tensor = torch.Tensor(np_image)
    
    if use_cuda:
        image_tensor = image_tensor.cuda()

    return image_tensor

def predict(image_path, model, topk, cat_to_name, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    use_cuda = gpu 
    # TODO: Implement the code to predict the class from an image file
    #preprocess image
    input_image = process_image(image_path, gpu)
    
    #check and display
    #input_image_tensor = input_image.cpu()
    #imshow(input_image_tensor)
    
    input_image = input_image.unsqueeze_(0)

    if use_cuda:
        model = model.cuda()
    
    model.eval()
    #make predictions
    output = model(input_image) 
    
    #Use SoftMax to get probabilities
    prob_output = F.softmax(output, dim=1).data
    
    # Use cpu to handle the numpy arrays
    prob_output = prob_output.cpu()

    # convert output probabilities to predicted class
    probs, top_i = prob_output.topk(topk)  
    top_i = top_i.numpy().squeeze()

    
    probs = probs.detach().numpy().squeeze()

    # transform to probabilities among the topk
    probs=probs/probs.sum()
    
    print(top_i)
    #cat_to_name[top_i]
    inverted_dict = {model.class_to_idx[class_element] : class_element for class_element in model.class_to_idx}
    
    if topk > 1:
        class_array = [inverted_dict[element] for element in top_i]
        names = [cat_to_name[str(element)] for element in class_array]
    else:
        print(type(top_i))
        class_array = inverted_dict[int(top_i)]
        names = cat_to_name[str(class_array[0])]
    
    #names = [cat_to_name[str(element)] for element in class_array]
    
    print("Most probable image classes and the respective probabilities")
    print()
    print(names)
    print()
    print(probs)
