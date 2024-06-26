from datasets import Dataset
import os
from PIL import Image as PILImage
from tqdm import tqdm

def load_dataset_with_images(data_dir, file_list_path, base_path, extension_input, extension_gt):
    with open(file_list_path, 'r') as f:
        file_paths = f.read().splitlines()
    
    images = []
    annotations = []
    
    for file_path in tqdm(file_paths):
        image_path = os.path.join(base_path, f"{file_path}{extension_input}")
        annotation_path = os.path.join(data_dir, f"{file_path}{extension_gt}")
        
        if os.path.exists(image_path) and os.path.exists(annotation_path):
            # Load and convert images to PIL.Image
            image = PILImage.open(image_path).convert("RGB")
            annotation = PILImage.open(annotation_path)
            
            images.append(image)
            annotations.append(annotation)
        else:
            print(f"File not found: {image_path} or {annotation_path}")
    
    data = {
        'image': images,
        'annotation': annotations,
    }
    dataset = Dataset.from_dict(data)
    return dataset

# path to directory containing annotated images
data_dir = "/data/datasets/1000_cityscapes_anotats_32400/annotated_images"
# path to file containing the paths of the annotated images
file_list_path = "cityscapes_train_1000.txt"
# base path to the images
base_path = '/data/datasets/cityscapes/leftImg8bit/train/'
# extensinos
extension_input = '_leftImg8bit.png'
extension_gt = '_gtFine_labelTrainIds.png'
# path to save the dataset
save_dir = "./cityscapes_train_1000_dataset_v3"

# Create the dataset
dataset = load_dataset_with_images(data_dir, file_list_path, base_path, extension_input, extension_gt)

# Save the dataset to disk
dataset.save_to_disk(save_dir)
print("Dataset saved successfully.")