import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("retail_sales_dataset.csv")
print(df.head())

df['Date'] = pd.to_datetime(df['Date'])
print(df.isnull().sum())  

category_sales = df.groupby('Product Category')['Total Amount'].sum()
category_sales.plot(kind='bar')
plt.title("Sales by Category")
plt.show()

df.groupby('Date')['Total Amount'].sum().plot()
plt.title("Sales Over Time")
plt.show()

sns.barplot(x='Gender', y='Total Amount', data=df)
plt.title("Gender vs Spending")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['Age', 'Quantity', 'Price per Unit']]
y = df['Total Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))