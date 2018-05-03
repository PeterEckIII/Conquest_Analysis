import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

file = "/Users/petereck/PycharmProjects/Conquest_Analysis/Data/Greater-New-York-2018-Conquest-Program-Ad-Sets-Lifetime (4).xlsx"

data = pd.read_excel(file)

df = pd.DataFrame(data)

x = df.iloc[:, :-1].values
y = df.iloc[:, :1].values

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(y_pred)

"""
data.plot(x="Amount Spent (USD)", y="Clicks (All)", style="o")
plt.title("Clicks Based on Media Spend")
plt.xlabel("Media Spend")
plt.ylabel("Total Clicks")
plt.show()
"""

