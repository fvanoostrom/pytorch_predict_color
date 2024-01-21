import torch
import json
import cv2
import numpy as np
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

def display_color_table(color_table):
    # Calculate the number of rows and columns in the grid
    num_colors = len(color_table)
    num_columns = int(np.ceil(np.sqrt(num_colors)))
    num_rows = int(np.ceil(num_colors / num_columns))

    # Set the size of each color box and the overall image
    box_size = 60
    image_width = num_columns * box_size
    image_height = num_rows * box_size

    # Create a blank white image
    image = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

    # Draw rectangles for each color with the color as the background and the name
    for i, color in enumerate(color_table):
        bgr_value = tuple(reversed(color["rgb_value"]))  # Convert RGB to BGR
        closest_bgr_value = tuple(reversed(color["closest_rgb_value"])) 
        closest_color_name = color["closest_color_name"]
        predicted_bgr_value = tuple(reversed(color["predicted_rgb_value"])) 
        predicted_color_name = color["predicted_color_name"]
        row, col = divmod(i, num_columns)
        y_start, y_end = row * box_size, (row + 1) * box_size
        x_start, x_end = col * box_size, (col + 1) * box_size
        text_color = (0, 0, 0) if predicted_color_name == closest_color_name else (0,0,255)

        # Top half with color_name and background color
        cv2.rectangle(image, (x_start, y_start), ((x_start + x_end) // 2, (y_start + y_end) // 2), bgr_value, -1)
        cv2.rectangle(image, ((x_start + x_end) // 2, y_start), (x_end, (y_start + y_end) // 2), closest_bgr_value, -1)
        cv2.putText(image, closest_color_name, (x_start + 5, (y_start + y_end) // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

        # Bottom half with "true_color" and background color
        cv2.rectangle(image, (x_start, (y_start + y_end) // 2), (x_end, y_end), predicted_bgr_value, -1)
        cv2.putText(image, predicted_color_name, (x_start + 5, y_end - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Color Table', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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
