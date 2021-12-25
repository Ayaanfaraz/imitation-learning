import random
import numpy as np

from collections import deque

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size # ()
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9    # discount rate
        self.loss_list = []
        self.epsilon = 0.21835392111705482  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995 # changed from 0.995, maybe it can be even slower 
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # OUR MODEL WILL BE RESNET WITH SOME LINEAR LAYERS AT THE END
        # INPUT = (1280, 720, 3) STACK OF 3 UNCERTAINTY MAPS
        # OUTPUT = (1, 13) 
        # ResNet 50
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(480, 640, 3))
        base_model.trainable = False
        # Additional Linear Layers
        inputs = keras.Input(shape=(480, 640, 3))
        #print(inputs)
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        # x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=40, activation='relu')(x)
        # x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(units=13, activation='linear')(x)
        # Compile the Model
        model = keras.Model(inputs, output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def memorize(self, state, action, reward, next_state, done):
        # print(state.shape, next_state.shape)
        self.memory.append((state, action, reward, next_state, done))

    def save_loss(self,loss):
        self.loss_list.append(loss)
        
    def act(self, state):
        # randomly select action
        if np.random.rand() <= self.epsilon:
            #[4-8] get random number
            return random.randrange(self.action_size)
            #return random.randint(5,7)
        # use NN to predict action
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state) # [[0.2, 0.3, 0.6, 0.1]]
        return np.argmax(act_values[0])  


    def replay(self, batch_size):
        # minibatch = random.sample(self.memory, batch_size)
        # states, targets_f = [], []
        # for state, action, reward, next_state, done in minibatch:
        #     state = np.expand_dims(state, axis=0)
        #     #print(state.shape)
        #     self.save_loss(np.amax(self.model.predict(state)[0]))
        #     # if done, set target = reward
        #     target = reward
        #     # if not done, predict future discounted reward with the Bellman equation
        #     #print("Before expand ", next_state.shape)
        #     next_state = np.expand_dims(next_state, axis=0)
        #     if not done:
        #         #print(np.amax(self.model.predict(next_state)[0])
        #         #print("Expanded ", next_state.shape)
        #         values = self.model.predict(next_state)
        #         target = (reward + self.gamma * np.amax(values[0]))
        #         #print("worked")

        #     state = np.expand_dims(state, axis=0)       
        #     target_f = self.model.predict(state)
        #     # print("my target_f is: ",target_f)
        #     #print("My target is: ",target)
        #     target_f[0][action] = target 
        #     # filtering out states and targets for training
        #     states.append(state[0])
        #     targets_f.append(target_f[0])

        #     ##TEST
        #     # history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        #     # loss = history.history['loss'][0]
        #     # if self.epsilon > self.epsilon_min:
        #     #     self.epsilon *= self.epsilon_decay
        #     # #print("Broke Here 2")
        #     # return loss
        #     ###TEST

        # # RUN ONE ITERATION OF GRADIENT DESCENT
        # print(len(states), len(targets_f))
        # history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # # Keeping track of loss
        # loss = history.history['loss'][0]
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        # #print("Broke Here 2")
        # return loss
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            # if done, set target = reward
            target = reward
            # if not done, predict future discounted reward with the Bellman equation
           # print("Before expand ", next_state.shape)
            next_state = np.expand_dims(next_state, axis=0)
            if not done:
                #print(np.amax(self.model.predict(next_state)[0])
              #  print("Expanded ", next_state.shape)
                values = self.model.predict(next_state)
                target = (reward + self.gamma * np.amax(values[0]))
             #   print("worked")

            state = np.expand_dims(state, axis=0)       
            target_f = self.model.predict(state)
            # print("my target_f is: ",target_f)
         #   print("My target is: ",target)
            target_f[0][action] = target 
            # filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        # RUN ONE ITERATION OF GRADIENT DESCENT
        print(len(states), len(targets_f))
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
      #  print("Broke Here 2")
        return loss