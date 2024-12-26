import torch
import cv2
import numpy as np
from Networks import build_advanced_segformer
from landslide_dataset import preprocess_image

# Settings
model_path = "best_model.pth"
input_channels = 14
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = build_advanced_segformer(num_classes=num_classes, num_input_channels=input_channels).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

def predict(image_path):
    image = preprocess_image(image_path, input_channels=input_channels).to(device)
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    return prediction

def visualize_predictions(image_path, mask_path):
    original_image = cv2.imread(image_path)  # Load original image for visualization
    predicted_mask = predict(image_path)

    # Create color map for visualization (modify as needed for your classes)
    color_map = {
        0: (0, 0, 0),      # Background (black)
        1: (255, 0, 0),    # Class 1 (red)
        2: (0, 255, 0),    # Class 2 (green)
        # Add more classes as needed...
    }

    vis_mask = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
    
    for class_id in color_map:
        vis_mask[predicted_mask == class_id] = color_map[class_id]

    # Overlay predicted mask on original image for visualization
    overlayed_image = cv2.addWeighted(original_image, 0.6, vis_mask, 0.4, 0)

    cv2.imshow('Original Image', original_image)
    cv2.imshow('Predicted Mask', vis_mask)
    cv2.imshow('Overlayed Image', overlayed_image)
    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_image_path = 'TestData/img/image_1.h5'   # Change this to your HDF5 file path.
    test_mask_path = 'TestData/mask/mask_1.h5'     # Change this to your mask file path.
    
    visualize_predictions(test_image_path, test_mask_path)
