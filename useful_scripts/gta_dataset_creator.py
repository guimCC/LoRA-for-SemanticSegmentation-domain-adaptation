import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import Dataset as HFDataset, DatasetDict

# NOTE: This batched way of creating and loading the dataset has been done because the dataset didn't fit directly on memory. Hence this approach.

class GTADataset(Dataset):
    def __init__(self, image_dir, label_dir, ids_list, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        with open(ids_list, 'r') as f:
            self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        # NOTE: Added sufixes to the image and label files 
        image = Image.open(os.path.join(self.image_dir, f"{img_id}.png")).convert("RGB")
        mask = Image.open(os.path.join(self.label_dir, f"{img_id}_labelTrainIds.png"))
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        image = np.array(image)
        mask = np.array(mask)
        
        return image, mask

def save_batches_to_disk(dataloader, save_dir, split_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    batch_num = 0
    for batch in dataloader:
        images, masks = batch
        batch_dataset = HFDataset.from_dict({
            'image': images.numpy(),
            'mask': masks.numpy()
        })
        
        batch_file = os.path.join(save_dir, f"{split_name}_batch_{batch_num}.arrow")
        batch_dataset.save_to_disk(batch_file)
        batch_num += 1

def load_gta_dataset(image_dir, label_dir, train_ids, val_ids, test_ids, batch_size=128, image_size=(512, 512)):
    print("Loading GTA dataset...")
    transform = lambda img: img.resize(image_size)
    
    train_dataset = GTADataset(image_dir, label_dir, train_ids, transform=transform)
    val_dataset = GTADataset(image_dir, label_dir, val_ids, transform=transform)
    test_dataset = GTADataset(image_dir, label_dir, test_ids, transform=transform)

    print("Datasets created.")

    # DataLoader for efficient batching
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    save_batches_to_disk(train_loader, save_path, 'train')
    save_batches_to_disk(val_loader, save_path, 'validation')
    save_batches_to_disk(test_loader, save_path, 'test')

    print("Batches saved to disk.")
    
    # Load batches back into DatasetDict
    def load_batches_from_disk(split_name):
        batches = []
        batch_num = 0
        while True:
            batch_file = os.path.join(save_path, f"{split_name}_batch_{batch_num}.arrow")
            if not os.path.exists(batch_file):
                break
            batch_dataset = HFDataset.load_from_disk(batch_file)
            batches.append(batch_dataset)
            batch_num += 1
        return HFDataset.concatenate(batches)

    hf_train = load_batches_from_disk('train')
    hf_val = load_batches_from_disk('validation')
    hf_test = load_batches_from_disk('test')

    return DatasetDict({
        'train': hf_train,
        'validation': hf_val,
        'test': hf_test
    })


# Path to GTA images converted to Cityscapes format
image_dir_ct = '/data/datasets/gta3/gta/images/LAB_translated/Cityscapes'
# Path to GTA labels converted to Cityscapes format
label_dir = '/data/datasets/gta3/gta/labels'
# .txt files containing the IDs of the images to be used in each split
train_ids = '/data/datasets/gta3/gta/gta_trainIds.txt'
val_ids = '/data/datasets/gta3/gta/gta_valIds.txt'
test_ids = '/data/datasets/gta3/gta/gta_testIds.txt'
# Directory to save the dataset
save_path = "./gta_dataset"

hf_datasets = load_gta_dataset(image_dir_ct, label_dir, train_ids, val_ids, test_ids, save_path)

hf_datasets.save_to_disk("./gta_dataset_final")
print("Final dataset saved successfully.")
