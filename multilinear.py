from Utils.DataManager import DataManager
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dm = DataManager()

x = dm.get_img_spatial_features()
print(x.shape)
y = dm.pt_500_a

lr = LinearRegression()
lr.fit(x.reshape(171,27),y.reshape(-1,1))

y_lr = lr.predict(x.reshape(171,27))

f1 = plt.figure()
plt.plot(x,y,'.',markersize=10, label='data')

plt.title(f"Multi-Linear Regression PT500 (MSE: {mean_squared_error(y, y_lr)})")

y2 = dm.pt_4000_a

lr2 = LinearRegression()
lr2.fit(x.reshape(171,27),y2.reshape(-1,1))

y2_lr = lr2.predict(x.reshape(171,27))

f2 = plt.figure()
plt.plot(x,y2,'.',markersize=10, label='data')

plt.title(f"Multi-Linear Regression PT4000 (MSE: {mean_squared_error(y2, y2_lr)})") 
plt.show()