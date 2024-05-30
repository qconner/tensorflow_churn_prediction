import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn.csv")
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

# X is for training, y is for testing predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#print(X_train.head())
#print(y_train.head())

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

model = load_model("tf_churn_model.keras")

yhat = model.predict(X_test)
yhat = [0 if val < 0.5 else 1 for val in yhat]
print(yhat)

print(accuracy_score(y_test, yhat))

