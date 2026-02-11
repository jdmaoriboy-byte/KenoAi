import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from art import text2art
from keras.callbacks import EarlyStopping
# Generate ASCII art with the text "KenoAi"
ascii_art = text2art("KenoAi")


print("============================================================")
print("KenoAi")
print("Created by: Corvus Codex")
print("Github: https://github.com/CorvusCodex/")
print("Licence : MIT License")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

print(ascii_art)
print("Keno prediction artificial intelligence")


# Load data from file, skipping header row
data = np.genfromtxt('data.csv', delimiter=',', dtype=int, skip_header=1)

# Replace all -1 values with 0
data[data == -1] = 0

# Prepare data to predict the next draw from the previous one
split = int(0.8 * len(data))
train_data, train_targets = data[:split-1], data[1:split]
val_data, val_targets = data[split-1:-1], data[split:]

max_value = 80
 
# Get the number of features from the data
num_features = train_data.shape[1]

model = keras.Sequential()
model.add(layers.Embedding(input_dim=max_value+1, output_dim=64))
model.add(layers.LSTM(512))
model.add(layers.Dense(num_features, activation='softmax'))  # Set the number of units to match the number of features

# Define the learning rate schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

# Use Adam optimizer with learning rate schedule
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)


# Compile the model before training
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
history = model.fit(train_data, train_targets, validation_data=(val_data, val_targets), epochs=300, callbacks=[EarlyStopping(patience=10, restore_best_weights=True)])

predictions = model.predict(val_data)

indices = np.argsort(predictions, axis=1)[:, -num_features:]
predicted_numbers = np.take_along_axis(val_data, indices, axis=1)


print("============================================================")
print("Predicted Numbers (10 per line):")
for numbers in predicted_numbers[:10]:
    print(', '.join(map(str, numbers[:10])))

print("============================================================")
print("If you won buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("Support my work:")
print("BTC: bc1q7wth254atug2p4v9j3krk9kauc0ehys2u8tgg3")
print("ETH & BNB: 0x68B6D33Ad1A3e0aFaDA60d6ADf8594601BE492F0")
print("Buy me a coffee: https://www.buymeacoffee.com/CorvusCodex")
print("============================================================")

# Prevent the window from closing immediately
input('Press ENTER to exit')
