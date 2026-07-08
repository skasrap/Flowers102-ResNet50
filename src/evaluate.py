import torch
import matplotlib.pyplot as plt
from collections import Counter

def evaluate_model(resnet, test_loader, device):
    num_sample = 0
    num_correct = 0
    preds = []
    true_labels = []
    
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = resnet(img) 
            _, predicted = torch.max(output, 1)
            _, true_label = torch.max(label, 1)
            
            preds += predicted.cpu().numpy().tolist()
            true_labels += true_label.cpu().numpy().tolist()
            
            num_sample += label.size(0)  
            num_correct += (predicted == true_label).sum().item()
            
        test_accuracy = 100 * num_correct / num_sample
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        
    return preds, true_labels, test_accuracy

def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, test_accuracy, preds, true_labels):
    # Plot train/val loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/Loss_plot.png')

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.axhline(y=test_accuracy, color='red', linestyle='--', label='Test Accuracy')
    plt.legend()
    plt.savefig('outputs/Acc_plot.png')
    plt.close()

    # Plot predictions vs true labels
    pred_count = Counter(preds)
    true_count = Counter(true_labels)
    
    p_values = pred_count.keys()
    p_counts = pred_count.values()
    plt.bar(p_values, p_counts, label='Predicted Labels')

    t_values = list(true_count.keys())
    t_counts = list(true_count.values())
    plt.bar(t_values, t_counts, alpha=0.25, color='red', label='True Labels')

    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Frequency of Predictions vs True Labels')
    plt.legend()
    plt.savefig('outputs/pred_vs_true_plot.png')
    plt.close()
