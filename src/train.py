import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

def train_model(resnet, train_loader, val_loader, device, epochs=50, lr=2e-3):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=lr) 

    for epoch in tqdm(range(epochs)): 
        resnet.train() 

        running_loss = 0.0
        num_samples = 0
        num_correct = 0
        
        for img, label in train_loader:
            img, label = img.to(device), label.to(device) 
            optimizer.zero_grad() 
            output = resnet(img)
            loss = loss_f(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)  
            _, true_label = torch.max(label, 1)
            num_samples += label.size(0)
            num_correct += (predicted == true_label).sum().item()
            
        train_accuracy = 100 * num_correct / num_samples
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        
        with torch.no_grad():
            vr_loss = 0
            val_samples = 0
            val_corrects = 0
            for img, label in val_loader:
                img, label = img.to(device), label.to(device)
                output = resnet(img)
                _, predicted = torch.max(output, 1)
                _, true_label = torch.max(label, 1)
                
                val_samples += label.size(0)
                val_corrects += (predicted == true_label).sum().item()
                vr_loss += loss_f(output, label).item()
                
            val_accuracy = 100 * val_corrects/val_samples
            val_loss = vr_loss/len(val_loader)
            
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
    return train_losses, train_accuracies, val_losses, val_accuracies
