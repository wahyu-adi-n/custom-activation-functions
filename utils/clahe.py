import os
import cv2
import numpy as np

root_path = "clahe_chest_xray/"

for dirpath, dirnames, filenames in os.walk(root_path):
    for file in filenames:
        image_path = os.path.join(dirpath, file)
        
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)
        
        cv2.imwrite(image_path, enhanced_image)
        
        