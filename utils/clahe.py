import os
import cv2
import numpy as np

root_path = "clahe_chest_xray/"

for dirpath, dirnames, filenames in os.walk(root_path):
    for file in filenames:
        image_path = os.path.join(dirpath, file)
        
        image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(image)
        

        # Adjust brightness
        alpha = 1.2  # Brightness factor (1.0 is neutral)
        adjusted_image = np.clip(alpha * enhanced_image, 0, 255).astype(np.uint8)
        cv2.imwrite(image_path, adjusted_image)
        
        