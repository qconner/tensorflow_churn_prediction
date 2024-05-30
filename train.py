import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("Churn.csv")

# one-hot encode the feature vector components
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))

# decode Churn indicator
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

# X is for training, y is for testing predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#print(X_train.head())
#print(y_train.head())

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

model = Sequential()

# for input layer,
# add dense, fully connected layer with 32 neurons
# activation func is linear 0 to 1 (Relu)
# ensure input dimension matches feature vector length
model.add(Dense(units=32, activation="relu", input_dim=len(X_train.columns)))

# add dense, fully connected layer with 32 neurons
model.add(Dense(units=64, activation="relu"))

# add single output, true or false to represent Churn predictioon
model.add(Dense(units=1, activation="sigmoid"))

# eval accuracy
print(model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"]))

print(model.fit(X_train, y_train, epochs=20, batch_size=32))


yhat = model.predict(X_test)
yhat = [0 if val < 0.5 else 1 for val in yhat]
print(yhat)

print(accuracy_score(y_test, yhat))

# save to directory
model.save("tf_churn_model.keras")
