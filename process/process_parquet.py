import json
import os

from datasets import Dataset, DatasetDict, Features
from datasets import Image as ImageFeature
from datasets import Value
from huggingface_hub import HfApi
from PIL import Image
from tqdm import tqdm


def convert_json_to_huggingface(json_path, image_base_dir=None):
    """
    Convert a JSON file to a Huggingface dataset.
    
    Args:
        json_path (str): Path to the JSON file.
        image_base_dir (str, optional): Base directory for images. If None, assumes image paths in JSON are absolute.
    
    Returns:
        Dataset: A Huggingface dataset.
    """
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Transform data structure
    transformed_data = []
    
    for item in tqdm(data):
        # Get image path
    # for i in range(100):
        # item = data[i]
        image_path = item['image']
        if image_base_dir:
            image_path = os.path.join(image_base_dir, image_path)
        
        # Ensure the image path exists
        if not os.path.exists(image_path):
            print(f"Warning: Image path does not exist: {image_path}")
            continue
        
        # Load image using PIL
        image = Image.open(image_path)
        
        # Take only the first answer
        answer = item['answers'][0]
        
        # Create new data structure
        transformed_item = {
            'image': image,
            'problem': item['question'],
            'solution': f'<answer> {answer} </answer>'
        }
        
        transformed_data.append(transformed_item)
    
    # Create Huggingface dataset
    features = Features({
        'image': ImageFeature(),
        'problem': Value('string'),
        'solution': Value('string')
    })
    
    dataset = Dataset.from_list(transformed_data, features=features)
    dataset_dict = DatasetDict({'train': dataset})
    return dataset_dict

def save_dataset(dataset, output_dir, to_hub=False, **kwargs):
    """
    Save the dataset to disk.
    
    Args:
        dataset (Dataset): Huggingface dataset to save.
        output_dir (str): Directory to save the dataset.
    """
    dataset.save_to_disk(output_dir)
    
    if to_hub:
        api = HfApi()
        api.create_repo(repo_id=repo_name, repo_type="dataset", exist_ok=True)
        
        # Upload the dataset
        dataset.push_to_hub(repo_name)
        print(f"Dataset pushed to https://huggingface.co/datasets/{repo_name}")
        
        print("\nTo load this dataset:")
        print(f"from datasets import load_dataset")
        print(f"dataset = load_dataset('{repo_name}')")
        print("data = dataset['train'][0]")
    
    
# push_to_hub(dataset_dict, "your-username/your-dataset-name", "your-hf-token")

# Example usage
if __name__ == "__main__":
    # Example paths (modify as needed)
    json_path = "playground/DocVQA/simplified_train.json"
    image_base_dir = "playground/DocVQA"  # Set to None if image paths are absolute
    output_dir = "playground/DocVQA-R1"
    
    # Convert data
    dataset = convert_json_to_huggingface(json_path, image_base_dir)
    
    # Save dataset
    save_dataset(dataset, output_dir)
    # save_dataset(dataset, output_dir, to_hub=True, repo_name="OCR_R1_test")