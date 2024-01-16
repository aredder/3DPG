import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.activations import gelu

class CriticNetwork(keras.Model):
    def __init__(self, agent_index, save_dir, layer1_dims=1024, layer2_dims=64, name='critic'):
        super(CriticNetwork, self).__init__()

        self.agent_index = agent_index
        self.save_dir = save_dir
        self.model_name = name + str(self.agent_index)
        self.save_file = os.path.join(self.save_dir, self.model_name + '_ddpg.tf')

        self.layer1 = Dense(layer1_dims, activation=gelu)
        self.layer2 = Dense(layer2_dims, activation=gelu)
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        x = self.layer1(tf.concat([state[:, 2:], action], axis=1))
        x = self.layer2(x)
        return self.q(x)

class CriticNetwork_4A(keras.Model):
    def __init__(self, agent_index, save_dir, layer1_dims=2048, layer2_dims=128, name='critic'):
        super(CriticNetwork_4A, self).__init__()

        self.agent_index = agent_index
        self.save_dir = save_dir
        self.model_name = name + str(self.agent_index)
        self.save_file = os.path.join(self.save_dir, self.model_name + '_ddpg.tf')

        self.layer1 = Dense(layer1_dims, activation=gelu)
        self.layer2 = Dense(layer2_dims, activation=gelu)
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        x = self.layer1(tf.concat([state[:, 0:14], state[:, 16:28], state[:, 32:42], state[:, 48:], action], axis=1))
        x = self.layer2(x)
        return self.q(x)


class ActorNetwork(keras.Model):
    def __init__(self, agent_index, save_dir, action_space, layer1_dims=64, layer2_dims=8, name='actor'):
        super(ActorNetwork, self).__init__()

        self.agent_index = agent_index
        self.save_dir = save_dir
        self.model_name = name + str(self.agent_index)
        self.save_file = os.path.join(self.save_dir, self.model_name + '_ddpg.tf')

        self.tower_11 = Dense(layer1_dims, activation=gelu)
        self.tower_21 = Dense(layer2_dims, activation=gelu)

        self.tower_12 = Dense(layer1_dims, activation=gelu)
        self.tower_22 = Dense(layer2_dims, activation=gelu)

        self.action1 = Dense(2, activation='tanh',
                             kernel_initializer=RandomUniform(minval=-10 ** (-2), maxval=10 ** (-2)))
        self.action2 = Dense(2, activation='tanh',
                             kernel_initializer=RandomUniform(minval=-10 ** (-2), maxval=10 ** (-2)))

        if self.agent_index == 1:
            self.tower_12.trainable = False
            self.tower_22.trainable = False
            self.action2.trainable = False
        else:
            self.tower_11.trainable = False
            self.tower_21.trainable = False
            self.action1.trainable = False

    def call(self, state):
        x1 = self.tower_11(state[:, 0:6])
        x2 = self.tower_12(state[:, 6:12])

        x1 = self.tower_21(x1)
        x2 = self.tower_22(x2)

        action1 = self.action1(x1)
        action2 = self.action2(x2)
        return tf.concat([action1, action2], axis=1)

# 4agent example

class ActorNetwork_4A(keras.Model):
    def __init__(self, agent_index, save_dir, action_space, layer1_dims=128, layer2_dims=16, name='actor'):
        super(ActorNetwork_4A, self).__init__()

        self.agent_index = agent_index
        self.save_dir = save_dir
        self.model_name = name + str(self.agent_index)
        self.save_file = os.path.join(self.save_dir, self.model_name + '_ddpg.tf')

        self.tower_11 = Dense(layer1_dims, activation=gelu)
        self.tower_21 = Dense(layer2_dims, activation=gelu)

        self.tower_12 = Dense(layer1_dims, activation=gelu)
        self.tower_22 = Dense(layer2_dims, activation=gelu)

        self.tower_13 = Dense(layer1_dims, activation=gelu)
        self.tower_23 = Dense(layer2_dims, activation=gelu)

        self.tower_14 = Dense(layer1_dims, activation=gelu)
        self.tower_24 = Dense(layer2_dims, activation=gelu)

        self.action1 = Dense(2, activation='tanh',
                             kernel_initializer=RandomUniform(minval=-10 ** (-2), maxval=10 ** (-2)))
        self.action2 = Dense(2, activation='tanh',
                             kernel_initializer=RandomUniform(minval=-10 ** (-2), maxval=10 ** (-2)))
        self.action3 = Dense(2, activation='tanh',
                             kernel_initializer=RandomUniform(minval=-10 ** (-2), maxval=10 ** (-2)))
        self.action4 = Dense(2, activation='tanh',
                             kernel_initializer=RandomUniform(minval=-10 ** (-2), maxval=10 ** (-2)))

        if self.agent_index == 1:
            self.tower_12.trainable = False
            self.tower_22.trainable = False
            self.action2.trainable = False

            self.tower_13.trainable = False
            self.tower_23.trainable = False
            self.action3.trainable = False

            self.tower_14.trainable = False
            self.tower_24.trainable = False
            self.action4.trainable = False
        elif self.agent_index == 2:
            self.tower_11.trainable = False
            self.tower_21.trainable = False
            self.action1.trainable = False

            self.tower_13.trainable = False
            self.tower_23.trainable = False
            self.action3.trainable = False

            self.tower_14.trainable = False
            self.tower_24.trainable = False
            self.action4.trainable = False
        elif self.agent_index == 3:
            self.tower_11.trainable = False
            self.tower_21.trainable = False
            self.action1.trainable = False

            self.tower_12.trainable = False
            self.tower_22.trainable = False
            self.action2.trainable = False

            self.tower_14.trainable = False
            self.tower_24.trainable = False
            self.action4.trainable = False
        elif self.agent_index == 4:
            self.tower_11.trainable = False
            self.tower_21.trainable = False
            self.action1.trainable = False

            self.tower_12.trainable = False
            self.tower_22.trainable = False
            self.action2.trainable = False

            self.tower_13.trainable = False
            self.tower_23.trainable = False
            self.action3.trainable = False

    def call(self, state):
        x1 = self.tower_11(state[:, 0:14])
        x2 = self.tower_12(state[:, 14:28])
        x3 = self.tower_12(state[:, 28:42])
        x4 = self.tower_12(state[:, 42:56])

        x1 = self.tower_21(x1)
        x2 = self.tower_22(x2)
        x3 = self.tower_22(x3)
        x4 = self.tower_22(x4)

        action1 = self.action1(x1)
        action2 = self.action2(x2)
        action3 = self.action2(x3)
        action4 = self.action2(x4)

        return tf.concat([action1, action2, action3, action4], axis=1)