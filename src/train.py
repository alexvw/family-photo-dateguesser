# train.py
import os
import torch
import torchvision
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torch import nn, optim

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'people'  # Directory where training data is located

    # Count the number of people by looking at the number of directories
    num_people = len([name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))])

    # Get the names of people
    people_names = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]

    # Load the pre-trained model
    model = models.resnet50(pretrained=True).to(device)

    # Change the last layer
    model.fc = nn.Linear(model.fc.in_features, num_people).to(device)

    # Prepare the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the data
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    # Prepare the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Fine-tune the model
    for epoch in range(10):  # You might need to increase the number of epochs
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(f"Device of inputs: {inputs.device}")
            print(f"Device of labels: {labels.device}")
            print(f"Device of model: {next(model.parameters()).device}")
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the model with a name containing the names of people
    model_name = '_'.join(people_names) + '.model'
    torch.save(model, model_name)

    return model_name, dataset.class_to_idx

if __name__ == "__main__":
    train_model()