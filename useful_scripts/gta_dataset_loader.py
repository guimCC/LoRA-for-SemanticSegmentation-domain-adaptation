import os
from datasets import concatenate_datasets, DatasetDict, load_from_disk

def load_batches(split_name, directory):
    batches = []
    batch_num = 0
    while True:
        batch_dir = os.path.join(directory, f"{split_name}_batch_{batch_num}.arrow")
        if not os.path.exists(batch_dir):
            break
        batch_dataset = load_from_disk(batch_dir)
        batches.append(batch_dataset)
        batch_num += 1
    return concatenate_datasets(batches) if batches else None

# Load each split
dataset_path = '../gta_dataset'

train_dataset = load_batches('train', dataset_path)
validation_dataset = load_batches('validation', dataset_path)
test_dataset = load_batches('test', dataset_path)

# Create a DatasetDict
hf_datasets = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

train_ds = hf_datasets["train"]
test_ds = hf_datasets["test"].train_test_split(test_size=0.1)['test']
val_ds = hf_datasets["validation"].train_test_split(test_size=0.1)['test']