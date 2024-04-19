from Utils.DataManager import DataManager
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dm = DataManager()

array = []
for img in dm.imgs:
    array.append(np.sum(img.flatten()))
x = np.array(array)
y = dm.pt_500_a # array, not tensor

lr = LinearRegression()
lr.fit(x.reshape(-1,1),y.reshape(-1,1))

y_lr = lr.predict(x.reshape(-1,1))

f1 = plt.figure()
plt.plot(x,y,'.',markersize=10, label='data')
plt.plot(x,y_lr, '-', label='linear regression') 

plt.legend()
plt.title(f"Linear Regression PT500 (MSE: {mean_squared_error(y, y_lr)})")

y2 = dm.pt_4000_a

lr2 = LinearRegression()
lr2.fit(x.reshape(-1,1),y2.reshape(-1,1))

y2_lr = lr2.predict(x.reshape(-1,1))

f2 = plt.figure()
plt.plot(x,y2,'.',markersize=10, label='data')
plt.plot(x,y2_lr, '-', label='linear regression') 

plt.legend()
plt.title(f"Linear Regression PT4000 (MSE: {mean_squared_error(y2, y2_lr)})") 
plt.show()


