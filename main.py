import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from DataGeneration import GenerateHAPTData, CSVDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from Pytorch_model import OneDCNN, LSTM, LSTM2, CNN_LSTM


def train_model(train_dl, model, epoch):
    # define loss function
    model = model.float()
    criterion = nn.CrossEntropyLoss()
    # define optimizer (you can try to change optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    # enumerate epochs
    accuracy = list()
    for epoch in range(epoch):
        predictions, actuals = list(), list()
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs.float())
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            # record result for mini batch
            predictions.append(np.argmax(yhat.detach().numpy(), axis=1))
            actuals.append(np.argmax(targets.numpy(), axis=1))
        predictions, actuals = np.concatenate(predictions), np.concatenate(actuals)
        acc = accuracy_score(actuals, predictions)
        accuracy.append(acc)
        print(f"Epoch: {epoch+1}; Accuracy: {acc}")

    # learning process visualization
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(accuracy)
    plt.show()


def model_evaluation(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        predictions.append(np.argmax(yhat.detach().numpy(), axis=1))
        actuals.append(np.argmax(targets.numpy(), axis=1))
    predictions, actuals = np.concatenate(predictions), np.concatenate(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    print(f"Test Accuracy: {acc}")


if __name__ == '__main__':
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # load data
    X, y = GenerateHAPTData().run()
    # transform data
    XT_lstm = torch.from_numpy(X).float()
    print(XT_lstm.shape)
    XT = XT_lstm.transpose(1,2).float() #input is (N, Cin, Lin) = Ntimesteps, Nfeatures, 128
    print(XT.shape)
    yT = torch.from_numpy(y).float()

    data = CSVDataset(XT, yT)
    data_lstm = CSVDataset(XT_lstm, yT)

    train, test = data.get_splits(train_rate=0.8)
    train_lstm, test_lstm = data_lstm.get_splits(train_rate=0.8)
    # create a data loader for train and test sets
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)

    train_lstm = DataLoader(train_lstm, batch_size=32, shuffle=True)
    test_lstm = DataLoader(test_lstm, batch_size=32, shuffle=False)

    n_timesteps = 128 #XT.shape[2]
    n_features = 6 #XT.shape[1]
    n_outputs = 12 #yT.shape[1]

    model = 2

    if model == 1:
        model = OneDCNN(n_timesteps, n_features, n_outputs)
        train_model(train_dl, model, epoch=15)
        model_evaluation(test_dl, model)

    if model == 2:
        model = LSTM(n_timesteps, n_features, n_outputs)
        train_model(train_lstm, model, epoch=15)
        model_evaluation(test_lstm, model)

    if model == 3:
        model = LSTM2(n_timesteps, n_features, n_outputs)
        train_model(train_lstm, model, epoch=10)
        model_evaluation(test_lstm, model)

    if model == 4:
        model = CNN_LSTM(n_timesteps, n_features, n_outputs)
        train_model(train_lstm, model, epoch=10)
        model_evaluation(test_lstm, model)

