# Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
# Date: 19-08-2025
# Developed By: A.Nabithra
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\admin\Downloads\Tesla.csv", parse_dates=['Date'], index_col='Date')

resampled_data = data['Close'].resample('Y').mean().to_frame()
resampled_data.index = resampled_data.index.year
resampled_data.reset_index(inplace=True)
resampled_data.rename(columns={'Date': 'Year', 'Close': 'ClosePrice'}, inplace=True)

years = resampled_data['Year'].tolist()
prices = resampled_data['ClosePrice'].tolist()

# A - LINEAR TREND ESTIMATION

X = [i - years[len(years) // 2] for i in years]
x2 = [i ** 2 for i in X]
xy = [i * j for i, j in zip(X, prices)]
n = len(years)

b = (n * sum(xy) - sum(prices) * sum(X)) / (n * sum(x2) - (sum(X) ** 2))
a = (sum(prices) - b * sum(X)) / n
linear_trend = [a + b * X[i] for i in range(n)]

# B- POLYNOMIAL TREND ESTIMATION

x3 = [i ** 3 for i in X]
x4 = [i ** 4 for i in X]
x2y = [i * j for i, j in zip(x2, prices)]

coeff = [[len(X), sum(X), sum(x2)],
         [sum(X), sum(x2), sum(x3)],
         [sum(x2), sum(x3), sum(x4)]]
Y = [sum(prices), sum(xy), sum(x2y)]
A = np.array(coeff)
B = np.array(Y)
solution = np.linalg.solve(A, B)
a_poly, b_poly, c_poly = solution

poly_trend = [a_poly + b_poly * X[i] + c_poly * (X[i] ** 2) for i in range(n)]

resampled_data['Linear Trend'] = linear_trend
resampled_data['Polynomial Trend'] = poly_trend
resampled_data.set_index('Year', inplace=True)


plt.figure(figsize=(8,5))
plt.plot(resampled_data.index, resampled_data['ClosePrice'], color='blue', marker='o', label='Close Price')
plt.plot(resampled_data.index, resampled_data['Linear Trend'], color='black', linestyle='--', label='Linear Trend')
plt.title("Tesla Stock Price - Linear Trend Estimation")
plt.xlabel("Year")
plt.ylabel("Average Close Price")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(resampled_data.index, resampled_data['ClosePrice'], color='blue', marker='o', label='Close Price')
plt.plot(resampled_data.index, resampled_data['Polynomial Trend'], color='red', marker='o', label='Polynomial Trend')
plt.title("Tesla Stock Price - Polynomial Trend Estimation")
plt.xlabel("Year")
plt.ylabel("Average Close Price")
plt.legend()
plt.grid(True)
plt.show()
```

### OUTPUT
```
Linear Trend: y=156.72 + 38.41x
Polynomial Trend: y=164.56 + 36.84x + -1.57x²
```

A - LINEAR TREND ESTIMATION

<img width="1206" height="743" alt="image" src="https://github.com/user-attachments/assets/5f971af9-e189-430b-980e-9e04de952311" />


B- POLYNOMIAL TREND ESTIMATION

<img width="1219" height="759" alt="image" src="https://github.com/user-attachments/assets/dabf6a49-8680-4345-9cad-a66fa60fa4ed" />

### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
