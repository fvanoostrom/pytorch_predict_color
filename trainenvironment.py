import torch

def train_color_model(model, train_dataloader, criterion, optimizer, num_epochs=50):
    # Train the neural network
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            inputs, targets = batch['input'], batch['target']

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
