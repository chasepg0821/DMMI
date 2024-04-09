from Utils.DataManager import DataManager
from Models.cnn3d import cnn3d
import torch
from sklearn.model_selection import train_test_split


epoch_number = 0
EPOCHS = 10

dm = DataManager()

X_train, X_test, y_train, y_test = train_test_split(dm.imgs, dm.get_pt_tensors(dataset=4000), test_size=0.33, random_state=42)

model = cnn3d()

train_data = [(X_train[i], y_train[i]) for i in range(0, len(X_train))]
test_data = [(X_test[i], y_test[i]) for i in range(0, len(X_test))]

training_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
validation_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        print(labels, outputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 12 == 11:
            last_loss = running_loss / 12 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))

    model.train()
    avg_loss = train_one_epoch()

    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    epoch_number += 1


