import torch
import torch.nn as nn   # neural network module
import torch.nn.functional as F   # functional module
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam   # optimization module
import torch.nn.init as initializers   # initialization module

import numpy as np
import matplotlib.pyplot as plt   # for plotting
import seaborn as sns   # for plotting

from pecnet.utils import Utility

class BasicNN(nn.Module):
    """
    A basic neural network class that extends PyTorch's nn.Module for creating and training neural network models.

    This class is designed to handle simple to moderately complex neural network tasks. It allows for the 
    initialization of a neural network with a customizable number of hidden layers and units, and includes 
    methods for training the network and making predictions. It is designed to be used with the PECNET Variable Network,
    Error Network and Final Network modules.

    Attributes:
        input_sequence_size (int): The size of the input sequence.
        output_sequence_size (int): The size of the output sequence.
        learning_rate (float): The learning rate for the optimizer.
        epoch_size (int): The number of epochs to train the network.
        batch_size (int): The batch size for training.
        hidden_layer_units_sizes (list): A list containing the sizes of each hidden layer.
        layers (nn.ModuleList): A ModuleList of the layers in the network.
        optimizer (torch.optim.Adam): The optimizer used for training.
        device (torch.device): The device (CPU/GPU) on which the network will run.

    Methods:
        __init__: Initializes the network and calls methods to initialize hyperparameters and available devices.
        init_hyperparameters: Initializes hyperparameters for the network.
        init_model: Initializes the model layers and optimizer.
        init_devices: Initializes the device (cpu or gpu) and moves model to this device.
        forward: Forwards the data through the network and gets predictions
        loss: Calculates the loss between predicted and target values.
        fit: Trains the model on input data.
        predict: Makes predictions on new data.
    """

    def __init__(self,sample_size,input_sequence_size,output_sequence_size):
        
        super().__init__()

        self.device = None

        self.init_hyperparameters(sample_size,input_sequence_size,output_sequence_size)
        self.init_model()
        self.init_devices()  

    def init_hyperparameters(self,sample_size,input_sequence_size,output_sequence_size):

        self.input_sequence_size = input_sequence_size
        self.output_sequence_size = output_sequence_size

        if Utility.heuristic:
            self.learning_rate = 0.01
            self.epoch_size = 300
            self.batch_size = int(np.sqrt(sample_size)) #The heuristic batch size parameter
            
            h1_hidden_units_size =int((2*input_sequence_size/3)+output_sequence_size) #The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer.
            h2_hidden_units_size =int(sample_size/(8*(input_sequence_size+output_sequence_size)))-h1_hidden_units_size #A kind of heuristic formula and 8 is a scale factor, which can be changed.     
            self.hidden_layer_units_sizes = [h1_hidden_units_size,h2_hidden_units_size] #There is currently no theoretical reason to use neural networks with any more than two hidden layers
        else:
            self.learning_rate = Utility.learning_rate
            self.epoch_size = Utility.epoch_size
            self.batch_size = Utility.batch_size
            self.hidden_layer_units_sizes = Utility.hidden_units_sizes

    def init_model(self):

        # Create the first layer from input to the first hidden layer
        layers=[nn.Linear(self.input_sequence_size, self.hidden_layer_units_sizes[0])]
        initializers.kaiming_normal_(layers[0].weight)  # He initialization for the first layer weights

        for i in range(1,len(self.hidden_layer_units_sizes)):
            layer=nn.Linear(self.hidden_layer_units_sizes[i-1], self.hidden_layer_units_sizes[i])
            initializers.kaiming_normal_(layer.weight)
            layers.append(layer)

        # Add the output layer
        layers.append(nn.Linear(self.hidden_layer_units_sizes[-1], self.output_sequence_size))
        initializers.kaiming_normal_(layers[-1].weight)

        self.layers = nn.ModuleList(layers)
        self.optimizer= Adam(self.parameters(), self.learning_rate)

    def init_devices(self):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.to(self.device)
        

        
    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.gelu(layer(x))
        
        # Apply the last layer without an activation function
        x = self.layers[-1](x)
        return x

    def loss(self, predicted, target):
        return torch.mean((predicted - target) ** 2)

    def fit(self, input_values, target_values):

        input_tensor = torch.tensor(input_values).float()
        output_tensor = torch.tensor(target_values).float()
        dataset = TensorDataset(input_tensor, output_tensor)
        train_loader=DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epoch_size):

            total_loss = 0.0  

            for inputs, targets in train_loader:
                
                inputs, targets = inputs.to(self.device), targets.to(self.device) 
                predict = self.forward(inputs)
                loss = self.loss(predict, targets)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
  
                total_loss += loss.item() * inputs.size(0)

            epoch_loss = total_loss / len(dataset)
            print(f"Epoch {epoch + 1}/{self.epoch_size}, Avg. Loss per Sample: {epoch_loss:.5f}")

            if total_loss<self.learning_rate/100:
                print(f"Stopping early at epoch {epoch+1} due to reaching threshold loss.")
                break
        
    def predict(self, X):
        
        self.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32, device=self.device)
            predictions = self(inputs)
            return predictions.detach().cpu().numpy()


if __name__ == "__main__":

    input_values=np.linspace(0, 2 * np.pi, 100).astype(np.float32).reshape(-1,1)
    target_values=np.sin(input_values).astype(np.float32).reshape(-1,1)

    sample_size=len(input_values)
    input_sequence_size=1
    output_sequence_size=1

    Utility.set_hyperparameters(learning_rate=0.001,
                                epoch_size=300,
                                batch_size=96,
                                hidden_units_sizes=[16,32,16,8])

    model=BasicNN(sample_size,input_sequence_size,output_sequence_size)
    model.fit(input_values,target_values)
    predictions=model.predict(input_values)

    print(predictions.shape)

    sns.set(style="whitegrid")
    sns.lineplot(x=input_values.flatten(), y=predictions.flatten(),color="green",linewidth=2.5)
    sns.lineplot(x=input_values.flatten(), y=target_values.flatten(),color="red",linewidth=2.5)
    plt.ylabel("Actual")
    plt.xlabel("X")
    plt.savefig("BasicSine.png")
