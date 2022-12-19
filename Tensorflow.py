import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from DataGeneration import GenerateHAPTData, CSVDataset
import numpy as np

X, y = GenerateHAPTData().run()
X = X.astype('float32')
Y = y.astype('float32')
x_tr, x_ts, y_tr, y_ts = train_test_split(X, Y, test_size=0.2, shuffle=True)


def create_model(X_train, y_train):
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]

    model = tf.keras.models.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(n_timesteps, n_features)),
        keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(n_outputs, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_lstm_model(X_train, y_train):
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]

    model = tf.keras.models.Sequential([
        keras.layers.LSTM(128, input_shape=(n_timesteps, n_features)),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(n_outputs, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def create_cnn_lstm_model(X_train, y_train):
    n_timesteps = X_train.shape[1]
    n_features = X_train.shape[2]
    n_outputs = y_train.shape[1]

    model = tf.keras.models.Sequential([
        keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=(n_timesteps, n_features)),
        keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.LSTM(128),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(n_outputs, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


model = create_model(x_tr, y_tr)
#model = create_lstm_model(x_tr, y_tr)
#model = create_cnn_lstm_model(x_tr, y_tr)

# graphical visualization
# keras.utils.plot_model(model, show_shapes=True)

# list summary
model.summary()

batch_size = 32
n_epochs = 15

train_history = model.fit(x_tr, y_tr, epochs=n_epochs, batch_size=batch_size, verbose=1)

#  training process visualization
# Fetch history
loss = train_history.history['loss']
accuracy = train_history.history['accuracy']

range_epochs = range(n_epochs)

plt.figure()
plt.subplot(1, 2, 1)
plt.plot(range_epochs, loss, 'r', label='Training loss')
plt.xlabel('Epoch')
plt.ylim([0, 1])
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(range_epochs, accuracy, 'b', label='Training accuracy')
plt.xlabel('Epoch')
plt.ylim([0, 1])
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(x_ts, y_ts, batch_size=batch_size, verbose=1)
print('\nTest accuracy:', test_acc)


# Create confusion matrix for tensorflow model

class_names = ['Walking',
          'WalkUp',
          'WalkDown',
          'Sit',
          'Stand',
          'Lay',
          'StandSit',
          'SitStand',
          'SitLie',
          'LieSit',
          'StandLie',
          'LieStand']


predictions = model.predict(x_ts)
predictions = np.argmax(predictions, axis=1)
labels = np.argmax(y_ts, axis=1)
cm = confusion_matrix(labels, predictions, normalize='true')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot()
plt.show()

