import torch
import json
from torch.utils.data import Dataset

def load_color_table(file_path):
    with open(file_path, 'r') as file:
        color_table = json.load(file)
        for color in color_table:
            
            color["rgb_value"] = [int(value) for value in color["rgb_value"]]
            color["unit_rgb"] = torch.tensor(color["rgb_value"], dtype=torch.float32) / 255.0
        return color_table

def create_color_table(unit_rgb_values, indexes, rgb_table, predicted_indexes = None):
    color_table = []
    predicted_indexes = indexes if predicted_indexes is None else predicted_indexes

    for i, (color, index, predicted_index) in enumerate(zip(unit_rgb_values,indexes, predicted_indexes)):
        color_table.append({
            "unit_rgb": color,
            "rgb_value": (color * 255).int().tolist(),
            "closest_color": indexes[i].item(),
            "closest_rgb_value": rgb_table[index]['rgb_value'],
            "closest_color_name": rgb_table[index]['color_name'],
            "predicted_color_name": rgb_table[predicted_index]['color_name'],
            "predicted_rgb_value": rgb_table[predicted_index]['rgb_value'],
        })
    # for color in color_table:
    #     print(f'rgb_value:{color["rgb_value"]}')
    #     print(f'closest_rgb_value:{color["closest_rgb_value"]}')
    #     print(f'predicted_rgb_value:{color["predicted_rgb_value"]}')
    return color_table
    
def generate_random_colors_with_index(quantity, color_table):
    # Generate multiple random colors in the range [0, 1]
    random_colors = torch.rand(quantity, 3)
    # Find the index of the closest color for each random color
    differences_rgb = torch.abs(color_table.unsqueeze(1) - random_colors)
    differences = torch.sum(differences_rgb, dim=2)
    nearest_indices = torch.argmin(differences, dim=0)

    # Return 
    # 1. the generated colors, 
    # 2. the absolute difference between the generated colors and a table of colors
    # 3. and the nearest color of that table
    return random_colors, differences, nearest_indices

class ColorDataset(Dataset):
    def __init__(self, input_data, target_data):
        self.input_data = input_data
        self.target_data = target_data

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]
        return {'input': input_sample, 'target': target_sample}
