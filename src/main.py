import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


data = pd.read_csv('./data/yield.csv')

# data.plot(kind='bar', x='day_length', y='yield')

# plt.show()

# data.describe()

# data.corr(numeric_only=True)

X = pd.DataFrame(data['day_length'])
Y = pd.DataFrame(data['yield'])

lm = linear_model.LinearRegression()

model = lm.fit(X,Y)

predicted_yield = model.predict(X)

fig = pd.DataFrame(predicted_yield, index=X['day_length'], columns=['yield'])

print("PV Power Prediction:")

print(fig)

print("PV Power Prediction Score: {}".format(model.score(X,Y)))

fig.columns = ['yield']

fig.plot(kind='bar')

plt.show()

if __name__ == '__main__':
    pass