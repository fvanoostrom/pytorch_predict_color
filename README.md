# Color Prediction Neural Network

This is solution is meant to demonstrate using pytorch with image data in the simplest form: to predict a color name based on the rgb values. The solution includes Python code to train a neural network for color prediction based on a color table. The neural network is trained to predict the closest color from the table for a given input color.

## Project Overview:

### Files:

1. **main.py**: The main script to initiate training. It loads the color table, creates datasets, initializes the neural network, and starts the training process.

2. **trainenvironment.py**: Contains the training environment function. It iterates over the specified number of epochs, updating the model parameters based on the calculated loss.

3. **dataset.py**: Provides functions for loading the color table, generating random colors with indexes, creating datasets, and displaying the color table.

4. **model.py**: Defines the neural network model using PyTorch. The `ColorPredictor` class is responsible for the architecture of the model.

### Dependencies:

- PyTorch
- Numpy
- OpenCV (cv2)

### How to Run:

This is a Python project. In order to run it, follow these steps:

1. If you haven't already, install Anaconda and Python. You can read how to do this on the [PyTorch website](https://pytorch.org/get-started/locally/).
1. Download CUDA. We make use of [version 12.1 of Cuda](https://developer.nvidia.com/cuda-12-1-0-download-archive).
1. Create (and activate) a new environment with Python 3.11.

   - **Linux** or **Mac**:
     ```bash
     conda create --name pytorch-image python=3.11
     source activate pytorch-image
     ```
   - **Windows**:
     ```bash
     conda create --name pytorch-image python=3.11 
     activate pytorch-image
     ```

1. Install the requirements in `requirements.txt` by typing the following command in the terminal:
    ```bash
    pip install -r requirements.txt
    ```

	-------

### Usage:

1. Run `main.py` to initiate the training process.
2. The script will load the color table from a JSON file, generate random colors, and train the neural network to predict the closest color from the table.
3. The trained model is then evaluated on a test set, and the accuracy is printed.
4. Finally, the color table with predicted values is displayed as an image.