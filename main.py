from trainenvironment import train_color_model
from dataset import load_color_table, create_color_table, generate_random_colors_with_index, display_color_table, ColorDataset
from model import ColorPredictor
import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    
    color_table_path = 'color_table.json'
    color_table_json = load_color_table(color_table_path)
    color_table = torch.stack([color["unit_rgb"] for color in color_table_json])

    # Create input and target tensors
    train_colors, train_differences, train_nearest_indices = generate_random_colors_with_index(5000, color_table)

    # Define neural network parameters
    input_size = 3
    hidden_size = 64
    output_size = len(color_table)

    # Initialize the neural network, loss function, and optimizer
    model = ColorPredictor(input_size, hidden_size, output_size)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create datasets and dataloaders
    train_dataset = ColorDataset(train_colors, train_nearest_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Call the training environment
    train_color_model(model, train_dataloader, criterion, optimizer, num_epochs=50)

    # Evaluate the trained model
    test_colors, test_differences, test_nearest_indices = generate_random_colors_with_index(100, color_table)
    model.eval()
    with torch.no_grad():
        predicted_indexes = torch.argmax(model(test_colors), dim=1)
    
    predicted_color_table = create_color_table(test_colors, test_nearest_indices, color_table_json, predicted_indexes)
    # Calculate the percentage of different values

    different_elements = torch.sum(predicted_indexes == test_nearest_indices).item()
    difference_percentage = (different_elements / predicted_indexes.numel()) * 100.0
    print(f'accuracy model:{difference_percentage} %')
    # Display the color table as an image
    display_color_table(predicted_color_table)
