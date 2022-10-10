import torch
import torch.nn as nn


class TurnLevelLSTM(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 lstm_dropout,
                 dropout_rate):
        super(TurnLevelLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=lstm_dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm2hiddnesize = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        lstm_out = self.lstm(inputs)
        lstm_out = lstm_out[0].squeeze(0)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.bilstm2hiddnesize(lstm_out)
        return lstm_out
