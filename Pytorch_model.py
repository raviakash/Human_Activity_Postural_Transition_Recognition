import torch.nn as nn


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
        return out


class LSTM(nn.Module):

    def __init__(self, n_timesteps, n_features, n_outputs):
        super(LSTM, self).__init__()
        self.layer2 = nn.LSTM(n_features, 64, 2, batch_first=True)
        self.layer3 = nn.Sequential(
            nn.Linear(64, n_outputs),
            nn.Softmax(dim=1))


    def forward(self, x):
        output, hidden = self.layer2(x)
        out = self.layer3(output)
        return out
    # def step(self, input, hidden=None):
    #     input = self.layer1(input).unsqueeze(1)
    #     output, hidden = self.layer2(input, hidden)
    #     output = self.layer3(output.squeeze(1))
    #     return output, hidden
    #
    # def forward(self, x, hidden=None):
    #     outputs = torch.zeros(32, 6, 12)
    #
    #     for i in range(6):
    #         input = x[:, i]
    #
    #         # print(input.shape)
    #         output, hidden = self.step(input, hidden)
    #
    #         # print(output.shape)
    #         outputs[:, i] = output
    #
    #     return outputs
