import pandas as pd
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
input_file = "data.csv"
df = pd.read_csv(input_file, header = 0)
# split into input (X) and output (y) variables
dataset = df.values
X = dataset[:,1:7]
y = dataset[:,7]
train_acc = 0
cross_acc = 0
for i in range(10):
    # split data into train, cross and test sets
    X_train, X_cross, y_train, y_cross = model_selection.train_test_split(X, y, train_size=.8, random_state=i)
    # define the Keras model
    model = Sequential()
    model.add(Dense(12, input_dim=6, activation='relu'))
    model.add(Dense(6, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the Keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the Keras model on the dataset
    model.fit(X_train, y_train, epochs=150, batch_size=5)
    # evaluate on training set
    _, accuracy = model.evaluate(X_train, y_train)
    train_acc = train_acc + accuracy
    # evaluate on cross set
    _, accuracy = model.evaluate(X_cross, y_cross)
    cross_acc = cross_acc + accuracy
train_acc = train_acc / 10
print('Train Accuracy: %.2f' % (train_acc * 100))
cross_acc = cross_acc / 10
print('Cross Accuracy: %.2f' % (cross_acc * 100))
# predicting with probability
predictions = model.predict(X_cross)
