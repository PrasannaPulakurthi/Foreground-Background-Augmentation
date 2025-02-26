import os
import yaml

# Define the root directory of the PACS dataset
root_dir = "PACS"

# Load the class mappings from the YAML file
yaml_file = os.path.join(root_dir, "PACS_categories.yaml")
with open(yaml_file, "r") as file:
    class_mapping = yaml.safe_load(file)

# Iterate through each domain in the PACS dataset
domains = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d != "PACS_categories.yaml"]

for domain in domains:
    domain_path = os.path.join(root_dir, domain)
    output_file = os.path.join(root_dir, f"{domain}_list.txt")
    
    with open(output_file, "w") as txt_file:
        # Traverse the domain directory
        for class_name in os.listdir(domain_path):
            class_path = os.path.join(domain_path, class_name)
            if os.path.isdir(class_path) and class_name in class_mapping:
                class_number = class_mapping[class_name]
                
                # Get all image files in the class folder
                for image_file in os.listdir(class_path):
                    if image_file.endswith(('.jpg', '.png', '.jpeg')):  # Filter for image files
                        file_path = os.path.join(domain, class_name, image_file)
                        file_path = file_path.replace("\\", "/")
                        txt_file.write(f"{file_path} {class_number}\n")

print("Text files generated for each domain!")
