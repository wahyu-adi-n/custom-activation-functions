from utils.data_preparation import data_loader_preparation
import numpy as np
import matplotlib.pyplot as plt

def show_samples(train_loader, num_classes, class_index):

    plt.figure(figsize=(20, 10))
    n_rows = 2
    n_cols = 5

    # Initialize counters for each class
    class_counters = {class_label: 0 for class_label in range(num_classes)}

    # Extract images from 'train_loader'
    images, labels = next(iter(train_loader))

    # Display the same 5 images for each class
    for class_label in range(num_classes):
        # Filter images for the current class
        class_images = images[labels == class_label][:5]

        # Display images
        for i in range(5):
            plt.subplot(n_rows, n_cols, class_label * 5 + i + 1)
            plt.imshow(np.transpose(class_images[i].numpy(), (1, 2, 0)))
            plt.title(class_index[class_label])
            plt.axis('off')

    plt.subplots_adjust(wspace=.02, hspace=-.2)
    plt.savefig("sample_images.jpg", format='jpg')
    plt.show()
    
if __name__ == '__main__':
    train_loader, _, _ = data_loader_preparation()
    show_samples(train_loader, 2, {0: 'NORMAL', 1: 'PNEUMONIA'})