import os
import pydicom
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from skimage import exposure
import torch
import torchvision.models as models
from torchvision import transforms
from sklearn.decomposition import PCA
import pandas as pd
from joblib import load  # For loading PCA model

# Load pre-trained ResNet-50 model for feature extraction
resnet_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])  # Remove FC layer
resnet_model.eval()

# Define image transformations for ResNet-50
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Move ResNet model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model.to(device)


def dicom_to_normalized_png(dicom_file, output_file):
    """Converts a single DICOM file to a normalized PNG image."""
    try:
        ds = pydicom.dcmread(dicom_file)
        img = ds.pixel_array

        img_normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        img_normalized = img_normalized.astype(np.uint8)

        img_equalized = exposure.equalize_hist(img_normalized)
        img_equalized_8bit = (img_equalized * 255).astype(np.uint8)

        Image.fromarray(img_equalized_8bit).convert("L").save(output_file)
        return output_file  # Return the path to the saved PNG
    except Exception as e:
        print(f"Error processing '{dicom_file}': {e}")
        return None


def extract_resnet_features(image_path):
    """Extracts features from a normalized PNG image using pre-trained ResNet-50."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = resnet_model(img_tensor)
            features = features.view(features.size(0), -1).cpu().numpy()

        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from '{image_path}': {e}")
        return None


def reduce_features_pca(features, n_components=256, pca_model_path='model/pca_model.joblib'):
    """Reduces the dimensionality of feature vectors using PCA."""
    try:
        if os.path.exists(pca_model_path):
            pca = load(pca_model_path)
            reduced_features = pca.transform(features.reshape(1, -1))
            return reduced_features.flatten()
        else:
            print(f"Warning: PCA model not found at '{pca_model_path}'.")
            return None
    except Exception as e:
        print(f"Error applying PCA: {e}")
        return None


def process_dicom_and_extract_features(dicom_file, output_dir="normalized_dicom_images", pca_components=256):
    """
    Wrapper function to process a DICOM file:
    - Convert to normalized PNG
    - Extract ResNet features
    - Apply PCA
    Returns:
    - torch.Tensor: reduced_features (PyTorch tensor)
    """
    os.makedirs(output_dir, exist_ok=True)

    if not dicom_file.endswith(".dcm"):
        print("Invalid input. Please provide a file with the '.dcm' extension.")
        return None

    base_name = os.path.basename(dicom_file).replace(".dcm", "")
    png_path = os.path.join(output_dir, f"{base_name}_normalized.png")

    # Step 1: Convert and normalize
    normalized_png_path = dicom_to_normalized_png(dicom_file, png_path)
    if not normalized_png_path:
        return None

    # Step 2: Extract ResNet features
    features = extract_resnet_features(normalized_png_path)
    if features is None:
        return None

    # Step 3: Reduce with PCA
    reduced_features = reduce_features_pca(features, n_components=pca_components)
    if reduced_features is None:
        return None

    return torch.tensor(reduced_features, dtype=torch.float32)


if __name__ == "__main__":
    dicom_input = input("Enter the path to the DICOM file you want to process: ")

    features_tensor = process_dicom_and_extract_features(dicom_input)

    if features_tensor is not None:
        print("\nFinal PCA-Reduced Feature Tensor (first 10 values):")
        print(features_tensor[:10])
        print(f"\nShape of final feature tensor: {features_tensor.shape}")
        # If needed, do: model(features_tensor.unsqueeze(0))