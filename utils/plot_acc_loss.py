import matplotlib.pyplot as plt

def plot_acc_loss_func(trai)
    plt.plot(train_loss_savings, label='Training Loss');
    plt.plot(val_loss_savings, label='Validation Loss');
    plt.title('Loss function');
    plt.xlabel('Epochs');
    plt.legend();

    # Accuracy
    plt.plot(train_acc_savings, label='Training Accuracy');
    plt.plot(val_acc_savings, label='Validation Accuracy');
    plt.title('Accuracy');
    plt.xlabel('Epochs');
    plt.legend();