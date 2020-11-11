import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data
import torch.nn as nn
from torch import optim
import os
import argparse

def preprocess_data(image_dir, batch_size = 30):
    
    train_dir = os.path.join(image_dir, "train")
    valid_dir = os.path.join(image_dir, "valid")
    test_dir = os.path.join(image_dir, "test")
    
    data_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p = 0.2),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform = data_transforms)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
    
    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform = test_transform)
    test_dataset = torchvision.datasets.ImageFolder(test_dir, transform = test_transform)
    
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = batch_size)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)
    
    class_to_idx = train_dataset.class_to_idx
    
    return trainloader, validloader, testloader, class_to_idx 
    
def build_model(hidden_units, arch):
    
    model = eval("models.{}(pretrained = True)".format(arch))
    for param in model.parameters():
        param.required_grad = False
    
    if arch == "densenet161":
        in_features = model.classifier.in_features
    elif arch == "alexnet":
        in_features = model.classifier[1].in_features
    else:
        in_features = model.classifier[0].in_features

    # Change the classifier
    model.classifier = nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(p = 0.25),
        nn.Linear(hidden_units, 500),
        nn.ReLU(),
        nn.Dropout(p = 0.25),
        nn.Linear(500, 102),
        nn.LogSoftmax(dim = 1))
    
    print("Built model")
    print("Printing Classifier")
    print(model.classifier)
    return model

def assess_model(test_loader, model, device):
    
    accuracy = 0
    
    model.eval()
    for image, label in test_loader:
        
        image, label = image.to(device), label.to(device)
        
        output = model(image)

        ps = torch.exp(model(image))
        top_prob, top_class = ps.topk(1, dim = 1)
        
        equality = top_class == label.view(*top_class.shape)
        accuracy = torch.mean(equality.type(torch.cuda.FloatTensor)).item()
        
        
    print("Accuracy:", str(accuracy))
    

def train(epochs, train_loader, valid_loader, test_loader, lr, device, class_to_idx, hidden_units, arch):
    
    applicable_models = ["alexnet", "densenet161", "vgg16"]
    if arch not in applicable_models:
        print("Please repick a base model")
        print(applicable_models)
        raise Exception("Model not in applicable models")
    
    model = build_model(hidden_units, arch)
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    
    for epoch in range(1, epochs + 1):
        train_loss = 0
        valid_loss = 0
        
        model.train()
        for batch_idx, (image, label) in enumerate(train_loader):
            
            image, label = image.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            
        model.eval()
        for image, label in valid_loader:
            
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            
            valid_loss += loss.item()
                
        print("train_loss:", str(train_loss))
        print("valid_loss:", str(valid_loss))
    
    model.class_to_idx = class_to_idx
    print("Training is finished")
    print("Model is being assesed")
    assess_model(test_loader, model, device)
    
    return model


def save_model(model, epochs, save_dir, learning_rate, base_arch, hidden_units):
    
    checkpoint = {"model": model.state_dict(),
                  "epochs": epochs,
                  "class_to_idx": model.class_to_idx,
                  "base_arch": base_arch,
                  "hidden_units": hidden_units}
    
    torch.save(checkpoint, save_dir)
    print("Model artifacts have been saved to directory:", str(save_dir))
    
def main():
    
    parser = argparse.ArgumentParser(description = "Runs training code for specified model")
    parser.add_argument("data_dir", type = str, help = "The data directory with the test, train, and valid folders")
    parser.add_argument("-a", "--arch", action = "store", type = str, help = "The torchvision model [alexnet, densenet161, vgg16] def: vgg16")
    parser.add_argument("-e", "--epochs", action = "store", type = int, help = "The amount of epochs to train for def: 10")
    parser.add_argument("-lr", "--learning_rate", action = "store", type = int, help = "The learning rate of model def: 0.001")
    parser.add_argument("-H", "--hidden_units", action = "store", type = int, help = "The amount of hidden units def: 1000")
    parser.add_argument("-s", "--save_dir", action = "store", type = str, help = "The additional path to save the model artifacts, def: in worspace")
    parser.add_argument("-g", "--gpu", action = "store_true", help = "Gives the ability to move to cuda if available.")
    
    args = parser.parse_args()
    
    image_dir = args.data_dir
    if args.arch:
        arch = args.arch
    else:
        arch = "vgg16"
    
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = 10
   
    if args.learning_rate:
        learning_rate = args.learning_rate
    else:
        learning_rate = 0.001
    
    if args.hidden_units:
        hidden_units = args.hidden_units
    else:
        hidden_units = 1000
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = "checkpoint.pth"
    
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("You are using:", str(device))
    else:
        device = torch.device("cpu")
        print("You are using cpu add -g to transport to cuda if available")
    
    trainloader, validloader, testloader, class_to_idx = preprocess_data(image_dir)
    trained_model = train(epochs, trainloader, validloader, testloader, learning_rate, device, class_to_idx, hidden_units, arch)
    save_model(trained_model, epochs, save_dir, learning_rate, arch, hidden_units)
            
if __name__ == "__main__":
    
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        