import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('insurance_data.csv')
print("First 5 rows of the dataset:")

print(df.head())
plt.scatter(df.age, df.bought_insurance, marker='+', color='blue')
plt.xlabel('Age')

plt.ylabel('Bought Insurance (0 = No, 1 = Yes)')
plt.title('Age vs Bought Insurance')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, train_size=0.8)

print("/nTest set (X_test):")
print(x_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

print("\nModel accuracy on test set:", model.score(x_test, y_test))
print("\nPredicted values (y_predicted):")

print(y_predicted)
print("\nActual values (y_test):")

print(y_test.values)
print("\nModel coefficient (m):", model.coef_)

print("\nModel intercept (b):", model.intercept_)

import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(age):
    z = 0.042 * age -1.53
    y = sigmoid(z)
    return y

age = 35
print(f"\nProbability that a person aged{age} buys insurance: {prediction_function(age):.3f}")

age= 43
print(f"\nProbability that a person aged {age} buys insurance: {prediction_function(age):.3f}")

plt.show()