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

# Change dm.pt-- for different dataset
X_train, X_test, y_train, y_test = train_test_split(dm.imgs, dm.pt_4000, test_size=0.33, random_state=42)

train_data = [(X_train[i], y_train[i]) for i in range(0, len(X_train))]
test_data = [(X_test[i], y_test[i]) for i in range(0, len(X_test))]

training_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)
 
model = cnn3d()

# MSE loss function, Adam optimizer with default CNN parameters
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
def train_one_epoch():
    total_batch_loss = 0.
    last_batch_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        # print("Labels: \n", labels, "\nOutputs: \n", outputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_batch_loss += loss.item()
        if i % 4 == 3:
            last_batch_loss = total_batch_loss / 4 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_batch_loss))
            total_batch_loss = 0.
    return last_batch_loss

# number of epochs to run
n_epochs = 100 

# Hold the best model (attempt to avoid overfitting by too many epochs)
best_mse = np.inf
best_weights = None
history = []
 
for epoch in range(n_epochs):
    model.train()
    # train
    with tqdm.tqdm(training_loader, unit="Batch") as bar:
        bar.set_description(f"Epoch {epoch + 1}")
        for data in bar:
            inputs, labels = data
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bar.set_postfix(mse=float(loss))
    # validate
    total_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            total_vloss += vloss
            avg_vloss = total_vloss / (i + 1)
        mse = float(avg_vloss)
        history.append(mse)
        # store best weights to avoid overfitting
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
 
# restore model and return best accuracy
model.load_state_dict(best_weights)
print("MSE: %.2f" % best_mse)
print("RMSE: %.2f" % np.sqrt(best_mse))
plt.plot(history)
plt.show()



