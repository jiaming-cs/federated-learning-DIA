from torch import nn
from torch.utils.data import DataLoader
INPUT_SIZE = 6 # num of feature for deep learning
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        
        self.lstm = nn.LSTM(
                input_size=INPUT_SIZE,
                hidden_size=32,
                num_layers=5,
                batch_first=True,
                )
        #fully connected
        self.out = nn.Linear(32, 2)
    
    def forward(self, x):
        lstm_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(lstm_out[:, -1, :])
        return out
