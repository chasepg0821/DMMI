from Utils.DataManager import DataManager
from Models.cnn3d import cnn3d
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import copy

dm = DataManager()

X_train, X_test, y_train, y_test = train_test_split(dm.imgs, dm.pt_500, test_size=0.33, random_state=42)

train_data = [(X_train[i], y_train[i]) for i in range(0, len(X_train))]
test_data = [(X_test[i], y_test[i]) for i in range(0, len(X_test))]

training_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)
 
model = cnn3d()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
n_epochs = 100   # number of epochs to run
 
def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        print("Labels: \n", labels, "\nOutputs: \n", outputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 12 == 11:
            last_loss = running_loss / 12 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

# Hold the best model
best_mse = np.inf   # init to infinity
best_weights = None
history = []
 
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(training_loader, unit="Batch") as bar:
        bar.set_description(f"Epoch {epoch + 1}")
        for data in bar:
            # take a batch
            inputs, labels = data
            # forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
            avg_vloss = running_vloss / (i + 1)
        mse = float(avg_vloss)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
 
# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()



