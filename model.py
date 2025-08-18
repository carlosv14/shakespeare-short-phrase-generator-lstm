import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, device=None):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, output_size)
        
        if device is not None:
            self.to(device)
    
    def forward(self, input, hidden):
        output, hidden = self.lstm(input, hidden)
        output = self.decoder(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device), 
                 torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device))
        return hidden