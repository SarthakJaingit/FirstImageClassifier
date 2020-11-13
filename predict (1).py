import torchvision
from torchvision import models, transforms
from PIL import Image
import argparse
import json
import torch.nn as nn
import torch.utils.data


def load_model(path, map_location):
    
    checkpoint = torch.load(path, map_location = map_location)
    model = eval("models.{}(pretrained = True)".format(checkpoint["base_arch"]))
    
    if checkpoint["base_arch"] == "densenet161":
        in_features = model.classifier.in_features
    elif checkpoint["base_arch"] == "alexnet":
        in_features = model.classifier[1].in_features
    else:
        in_features = model.classifier[0].in_features
    
    model.classifier = nn.Sequential(
        nn.Linear(in_features, checkpoint["hidden_units"]),
        nn.ReLU(),
        nn.Dropout(p = 0.25),
        nn.Linear(checkpoint["hidden_units"], 500),
        nn.ReLU(),
        nn.Dropout(p = 0.25),
        nn.Linear(500, 102),
        nn.LogSoftmax(dim = 1))
    
    model.load_state_dict(checkpoint["model"])
    model.class_to_idx = checkpoint["class_to_idx"]
    epochs = checkpoint["epochs"]
    
    return model

def preprocess_image(image_dir):
    
    image = Image.open(image_dir)
    rgb_image = image.convert("RGB")
    
    data_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    rgb_image = data_transforms(rgb_image)
    return rgb_image

def predict(image_path, model, topk, json_mapping, device):
    
    rgb_image = preprocess_image(image_path).unsqueeze_(0)
    model.to(device)
    rgb_image = rgb_image.to(device)
    
    output = model(rgb_image)
    prob = torch.exp(output)
    top_prob, top_class = prob.topk(topk, dim = 1)
    
    idx_to_class = {model.class_to_idx[folder_type]: folder_type for folder_type in model.class_to_idx}
    top_classes_fn = [idx_to_class[index.item()] for index in top_class[0]]
    
    with open(json_mapping, 'r') as f:
        json_data = json.load(f)
    
    if len(json_data) != 102:
        raise Exception("Please have category.json have 102 elements")
    
    real_name = [json_data[folder_name] for folder_name in top_classes_fn]
    
    print("Top Probabilities for topk: {}".format(top_prob))
    print("Top classes folder name: {}".format(top_classes_fn))
    print("Predicted classes: {}".format(real_name))
    print("Best prediction: {}".format(real_name[0]))
    

def main():
    
    parser = argparse.ArgumentParser(description = "Command Line for predict.py")
    
    parser.add_argument("image_dir", type = str, help = "This is directory to the image you want to predict")
    parser.add_argument("checkpoint", type = str, help = "This is the directory where you stored a pretrained model artifact")
    parser.add_argument("-tk", "--topk", type = int, action = "store", help = "This is where you define topk, def: 5")
    parser.add_argument("-c", "--category_names", type = str, action = "store", help = "directory to category mapping, def: cat_to_names.json direct")
    parser.add_argument("-g", "--gpu", action = "store_true", help = "Gives the ability to move to cuda if available.")
    
    args = parser.parse_args()
    
    if args.topk:
        topk = args.topk
    else:
        topk = 5
    
    if args.category_names:
        json_mapping = args.category_names
    else:
        json_mapping = "cat_to_name.json"
        
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("You are using:", str(device))
    else:
        device = torch.device("cpu")
        print("You are using cpu add -g to transport to cuda if available")
    
    if device == "cuda":
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    
    print("Map Location {}".format(map_location))
    model = load_model("saved_models/checkpoint.pth", map_location)
    print("Loaded model")
    predict(args.image_dir, model, topk, json_mapping, device)
        
if __name__ == "__main__":
    
    main()
    



