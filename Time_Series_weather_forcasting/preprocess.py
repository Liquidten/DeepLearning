import os
import numpy as np

# inspecting the data
data_dir = '/Users/sshah23/Documents/Deep_Learning_Data/jena_climate'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

# parsing the data
float_data = np.zeros((len(lines), len(header) -1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# ploting the temp data for the past 10 days
# from matplotlib import pyplot as plt
#
# temp = float_data[:, 1]
# #plt.plot(range(len(temp)), temp)
# plt.plot(range(1440), temp[:1440])
# plt.show()
#print(float_data)

# Normalizing the data
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

# generator yielding timeseries samples and their targets
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128,
              step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        sample = np.zeros((len(rows), lookback // step, data.shape[-1]))
        target = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback,  rows[j], step)
            target[j] = data[rows[j] + delay][1]
        yield sample, target


# preparing training validation and test data
lookback  = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data, lookback=lookback, delay=delay,min_index=0,
                      max_index=200000, shuffle=True, step=step, batch_size=batch_size)

val_gen = generator(float_data, lookback=lookback, delay=delay,min_index=200001,
                      max_index=300000, shuffle=True, step=step, batch_size=batch_size)

test_gen = generator(float_data, lookback=lookback, delay=delay,min_index=300001,
                      max_index=None, shuffle=True, step=step, batch_size=batch_size)

val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)

# Computing the common-sence baseline MAE
def evaluate_naive_method():
    batch_mae = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1 , 1]
        mae = np.mean(np.abs(preds - targets))
        batch_mae.append(mae)
    print(np.mean(batch_mae))

evaluate_naive_method()
celsius_mae = 0.76 * std[1]
print(celsius_mae)

# Training and evaluating a desely connected model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


model = tf.keras.Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop', loss='mae')
history = model.fit_generator(train_gen)
