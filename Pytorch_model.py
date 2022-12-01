import torch.nn as nn
import torch

class OneDCNN(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(OneDCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Sequential(
            nn.Linear(3904,100),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(100,n_outputs),
            nn.Softmax(dim=1))
       
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #print(out.shape)
        return out


class LSTM(nn.Module):

    def __init__(self, n_timesteps, n_features, n_outputs):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(6, 128, 2, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(128, n_outputs),
            nn.Softmax(dim=1))

    def forward(self, x):
        output, hidden = self.lstm(x)
        #print(output.shape)
        output = output[:, 127, :]
        #print(output.shape)
        out = self.out(output.squeeze(1))
        return out

class CNN_LSTM(nn.Module):

    def __init__(self, n_timesteps, n_features, n_outputs):
        super(CNN_LSTM, self).__init__()

        self.lstm = nn.LSTM(6, 128, 2, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(128, n_outputs),
            nn.Softmax(dim=1))

    def forward(self, x):
        output, hidden = self.lstm(x)
        #print(output.shape)
        output = output[:, 127, :]
        #print(output.shape)
        out = self.out(output.squeeze(1))
        return out


class LSTM2(nn.Module):

    def __init__(self, n_timesteps, n_features, n_outputs):
        super(LSTM2, self).__init__()

        self.inp = nn.Linear(n_features, 128)
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(128, n_outputs),
            nn.Softmax(dim=1))

    def step(self, input, hidden=None):
        input = self.inp(input).unsqueeze(1)
        output, hidden = self.lstm(input, hidden)
        output = self.out(output.squeeze(1))

        return output, hidden

    def forward(self, x, hidden=None):
        outputs = torch.zeros(x.shape[0], x.shape[1], 12)

        for i in range(128):
            input = x[:, i]

            # print(input.shape)
            output, hidden = self.step(input, hidden)

            # print(output.shape)
            outputs[:, i] = output

        return output

