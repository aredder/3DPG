import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

from .tools import ReplayBuffer
from .agent_networks import ActorNetwork, CriticNetwork, ActorNetwork_4A, CriticNetwork_4A


class LRSchedule_Actor(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
     return self.initial_learning_rate / (step/1000 + 1)


class LRSchedule_Critic(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
     return self.initial_learning_rate / (step/1000 + 1) + self.initial_learning_rate/(step/1000 + 1)**2


class Agent:
    def __init__(self, agent_index, save_dir, state_space, action_space, critic_lr=np.exp(-6), actor_lr=np.exp(-6),
                 alpha=0.9, max_size=20000, tau=0.01, batch_size=128, train_critic=True):

        self.train_critic = train_critic
        self.agent_index = agent_index
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha    # Discount factor
        self.tau = tau
        self.memory = ReplayBuffer(max_size, state_space, action_space)
        self.batch_size = batch_size

        self.critic = CriticNetwork(agent_index=agent_index, save_dir=save_dir)
        self.actor = ActorNetwork(agent_index=agent_index, save_dir=save_dir, action_space=action_space)
        self.target_critic = CriticNetwork(agent_index=agent_index, save_dir=save_dir)
        self.target_actor = ActorNetwork(agent_index=agent_index, save_dir=save_dir, action_space=action_space)

        self.actor.compile(optimizer=Adam(learning_rate=LRSchedule_Actor(actor_lr)))
        self.critic.compile(optimizer=Adam(learning_rate=LRSchedule_Critic(critic_lr)))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        # Initialise targets
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_networks(self):
        if self.memory.mem_cntr > 10*self.batch_size:
            if self.train_critic:
                weights = []
                targets = self.target_actor.weights
                for i, weight in enumerate(self.actor.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_actor.set_weights(weights)

                weights = []
                targets = self.target_critic.weights
                for i, weight in enumerate(self.critic.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        self.actor.save_weights(self.actor.save_file, save_format='tf')
        self.critic.save_weights(self.critic.save_file, save_format='tf')

    def load_models(self):
        self.actor.load_weights(self.actor.save_file).expect_partial()
        self.critic.load_weights(self.critic.save_file).expect_partial()

    def pick_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        return actions[0]

    def train(self):

        if self.memory.mem_cntr < 10*self.batch_size:
            return 0

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        # rewards = tf.convert_to_tensor(reward, dtype=tf.float32)  # not becessary
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        if self.train_critic:
            with tf.GradientTape() as tape:
                target_actions = self.target_actor(states_)
                critic_value_ = tf.squeeze(self.target_critic(
                    states_, target_actions), 1)
                critic_value = tf.squeeze(self.critic(states, actions), 1)
                target = reward + self.alpha * critic_value_ * (1 - done)
                critic_loss = keras.losses.MSE(target, critic_value)

            critic_network_gradient = tape.gradient(critic_loss,
                                                    self.critic.trainable_variables)
            for grad in critic_network_gradient:
                if tf.math.count_nonzero(grad) == 0:
                    print('Critic gradient dead')

            self.critic.optimizer.apply_gradients(zip(
                critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)

        for grad in actor_network_gradient:
            if tf.math.count_nonzero(grad) == 0:
                print('Actor gradient dead')

        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_target_networks()

        if self.train_critic:
            return critic_loss
        else:
            return 1

    def update_actor_weights_by_name(self, layer_name, weights):
        self.actor.model.get_layer(layer_name).set_weights(weights)


class Agent_4A:
    def __init__(self, agent_index, save_dir, state_space, action_space, critic_lr=np.exp(-6), actor_lr=np.exp(-6),
                 alpha=0.9, max_size=30000, tau=0.01, batch_size=256, train_critic=True):

        self.train_critic = train_critic
        self.agent_index = agent_index
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha    # Discount factor
        self.tau = tau
        self.memory = ReplayBuffer(max_size, state_space, action_space)
        self.batch_size = batch_size

        self.critic = CriticNetwork_4A(agent_index=agent_index, save_dir=save_dir)
        self.actor = ActorNetwork_4A(agent_index=agent_index, save_dir=save_dir, action_space=action_space)
        self.target_critic = CriticNetwork_4A(agent_index=agent_index, save_dir=save_dir)
        self.target_actor = ActorNetwork_4A(agent_index=agent_index, save_dir=save_dir, action_space=action_space)

        self.actor.compile(optimizer=Adam(learning_rate=LRSchedule_Actor(actor_lr)))
        self.critic.compile(optimizer=Adam(learning_rate=LRSchedule_Critic(critic_lr)))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        # Initialise targets
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_networks(self):
        if self.memory.mem_cntr > 10*self.batch_size:
            if self.train_critic:
                weights = []
                targets = self.target_actor.weights
                for i, weight in enumerate(self.actor.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_actor.set_weights(weights)

                weights = []
                targets = self.target_critic.weights
                for i, weight in enumerate(self.critic.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        self.actor.save_weights(self.actor.save_file, save_format='tf')
        self.critic.save_weights(self.critic.save_file, save_format='tf')

    def load_models(self):
        self.actor.load_weights(self.actor.save_file).expect_partial()
        self.critic.load_weights(self.critic.save_file).expect_partial()

    def pick_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        return actions[0]

    def train(self):

        if self.memory.mem_cntr < 10*self.batch_size:
            return 0

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        # rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        if self.train_critic:
            with tf.GradientTape() as tape:
                target_actions = self.target_actor(states_)
                critic_value_ = tf.squeeze(self.target_critic(
                    states_, target_actions), 1)
                critic_value = tf.squeeze(self.critic(states, actions), 1)
                target = reward + self.alpha * critic_value_ * (1 - done)
                critic_loss = keras.losses.MSE(target, critic_value)

            critic_network_gradient = tape.gradient(critic_loss,
                                                    self.critic.trainable_variables)
            for grad in critic_network_gradient:
                if tf.math.count_nonzero(grad) == 0:
                    print('Critic gradient dead')

            self.critic.optimizer.apply_gradients(zip(
                critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)

        for grad in actor_network_gradient:
            if tf.math.count_nonzero(grad) == 0:
                print('Actor gradient dead')

        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_target_networks()

        if self.train_critic:
            return critic_loss
        else:
            return 1

    def update_actor_weights_by_name(self, layer_name, weights):
        self.actor.model.get_layer(layer_name).set_weights(weights)


class Agent_Lowe:
    def __init__(self, agent_index, save_dir, state_space, action_space, critic_lr=np.exp(-6), actor_lr=np.exp(-6),
                 alpha=0.9, max_size=20000, tau=0.01, batch_size=128, train_critic=True):

        self.train_critic = train_critic
        self.agent_index = agent_index
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha    # Discount factor
        self.tau = tau
        self.memory = ReplayBuffer(max_size, state_space, action_space)
        self.batch_size = batch_size

        self.critic = CriticNetwork(agent_index=agent_index, save_dir=save_dir)
        self.actor = ActorNetwork(agent_index=agent_index, save_dir=save_dir, action_space=action_space)
        self.target_critic = CriticNetwork(agent_index=agent_index, save_dir=save_dir)
        self.target_actor = ActorNetwork(agent_index=agent_index, save_dir=save_dir, action_space=action_space)

        self.actor.compile(optimizer=Adam(learning_rate=LRSchedule_Actor(actor_lr)))
        self.critic.compile(optimizer=Adam(learning_rate=LRSchedule_Critic(critic_lr)))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        # Initialise targets
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_networks(self):
        if self.memory.mem_cntr > 10*self.batch_size:
            if self.train_critic:
                weights = []
                targets = self.target_actor.weights
                for i, weight in enumerate(self.actor.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_actor.set_weights(weights)

                weights = []
                targets = self.target_critic.weights
                for i, weight in enumerate(self.critic.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        self.actor.save_weights(self.actor.save_file, save_format='tf')
        self.critic.save_weights(self.critic.save_file, save_format='tf')

    def load_models(self):
        self.actor.load_weights(self.actor.save_file).expect_partial()
        self.critic.load_weights(self.critic.save_file).expect_partial()

    def pick_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        return actions[0]

    def train(self):
        """ Original code by Phil Tabor """

        if self.memory.mem_cntr < 10*self.batch_size:
            return 0

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        # rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        if self.train_critic:
            with tf.GradientTape() as tape:
                target_actions = self.target_actor(states_)
                critic_value_ = tf.squeeze(self.target_critic(
                    states_, target_actions), 1)
                critic_value = tf.squeeze(self.critic(states, actions), 1)
                target = reward + self.alpha * critic_value_ * (1 - done)
                critic_loss = keras.losses.MSE(target, critic_value)

            critic_network_gradient = tape.gradient(critic_loss,
                                                    self.critic.trainable_variables)
            for grad in critic_network_gradient:
                if tf.math.count_nonzero(grad) == 0:
                    print('Critic gradient dead')

            self.critic.optimizer.apply_gradients(zip(
                critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            if self.agent_index == 1:
                sampled_actions = tf.concat([self.actor(states)[:, 0:2], actions[:, 2:4]], 1)

            elif self.agent_index == 2:
                sampled_actions = tf.concat([actions[:, 0:2], self.actor(states)[:, 2:4]], 1)
            actor_loss = -self.critic(states, sampled_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)

        for grad in actor_network_gradient:
            if tf.math.count_nonzero(grad) == 0:
                print('Actor gradient dead')

        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_target_networks()

        if self.train_critic:
            return critic_loss


    def update_actor_weights_by_name(self, layer_name, weights):
        self.actor.model.get_layer(layer_name).set_weights(weights)


class Agent_Lowe_4A:
    def __init__(self, agent_index, save_dir, state_space, action_space, critic_lr=np.exp(-6), actor_lr=np.exp(-6),
                 alpha=0.9, max_size=30000, tau=0.01, batch_size=256, train_critic=True):

        self.train_critic = train_critic
        self.agent_index = agent_index
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha    # Discount factor
        self.tau = tau
        self.memory = ReplayBuffer(max_size, state_space, action_space)
        self.batch_size = batch_size

        self.critic = CriticNetwork_4A(agent_index=agent_index, save_dir=save_dir)
        self.actor = ActorNetwork_4A(agent_index=agent_index, save_dir=save_dir, action_space=action_space)
        self.target_critic = CriticNetwork_4A(agent_index=agent_index, save_dir=save_dir)
        self.target_actor = ActorNetwork_4A(agent_index=agent_index, save_dir=save_dir, action_space=action_space)

        self.actor.compile(optimizer=Adam(learning_rate=LRSchedule_Actor(actor_lr)))
        self.critic.compile(optimizer=Adam(learning_rate=LRSchedule_Critic(critic_lr)))
        self.target_actor.compile(optimizer=Adam(learning_rate=actor_lr))
        self.target_critic.compile(optimizer=Adam(learning_rate=critic_lr))

        # Initialise targets
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_networks(self):
        if self.memory.mem_cntr > 10*self.batch_size:
            if self.train_critic:
                weights = []
                targets = self.target_actor.weights
                for i, weight in enumerate(self.actor.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_actor.set_weights(weights)

                weights = []
                targets = self.target_critic.weights
                for i, weight in enumerate(self.critic.weights):
                    weights.append(weight * self.tau + targets[i] * (1 - self.tau))
                self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        self.actor.save_weights(self.actor.save_file, save_format='tf')
        self.critic.save_weights(self.critic.save_file, save_format='tf')

    def load_models(self):
        self.actor.load_weights(self.actor.save_file).expect_partial()
        self.critic.load_weights(self.critic.save_file).expect_partial()

    def pick_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        return actions[0]

    def train(self):
        """ Original code by Phil Tabor """

        if self.memory.mem_cntr < 10*self.batch_size:
            return 0

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        # rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        if self.train_critic:
            with tf.GradientTape() as tape:
                target_actions = self.target_actor(states_)
                critic_value_ = tf.squeeze(self.target_critic(
                    states_, target_actions), 1)
                critic_value = tf.squeeze(self.critic(states, actions), 1)
                target = reward + self.alpha * critic_value_ * (1 - done)
                critic_loss = keras.losses.MSE(target, critic_value)

            critic_network_gradient = tape.gradient(critic_loss,
                                                    self.critic.trainable_variables)
            for grad in critic_network_gradient:
                if tf.math.count_nonzero(grad) == 0:
                    print('Critic gradient dead')

            self.critic.optimizer.apply_gradients(zip(
                critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            if self.agent_index == 1:
                sampled_actions = tf.concat([self.actor(states)[:, 0:2], actions[:, 2:8]], 1)

            elif self.agent_index == 2:
                sampled_actions = tf.concat([actions[:, 0:2], self.actor(states)[:, 2:4], actions[:, 4:8]], 1)

            elif self.agent_index == 3:
                sampled_actions = tf.concat([actions[:, 0:4], self.actor(states)[:, 4:6], actions[:, 6:8]], 1)

            elif self.agent_index == 4:
                sampled_actions = tf.concat([self.actor(states)[:, 0:6], actions[:, 6:8]], 1)

            actor_loss = -self.critic(states, sampled_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)

        for grad in actor_network_gradient:
            if tf.math.count_nonzero(grad) == 0:
                print('Actor gradient dead')

        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_target_networks()

        if self.train_critic:
            return critic_loss


    def update_actor_weights_by_name(self, layer_name, weights):
        self.actor.model.get_layer(layer_name).set_weights(weights)