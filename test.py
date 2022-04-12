import gym

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import losses

import numpy as np

learning_rate = 0.01
gamma = 0.99
epsilon = 0.01



class DQN(tf.keras.Model):
    def __init__(self, lr, decay, epsilon, n_actions, n_features):
        super(DQN, self).__init__()
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self.actions = n_actions
        self.features = n_features

        self.build_net()



    def build_net(self):
        self.d1 = Dense(24, activation=tf.nn.relu)
        self.d2 = Dense(24, activation=tf.nn.relu)
        self.d3 = Dense(2)

    def call(self, x):
        # print(x)

        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)


 

    def choose_action(self, inputs):
        q_eval = self.call(inputs)
        return tf.argmax(input=q_eval)




if __name__=='__main__':
    env = gym.make('CartPole-v1')
    n_actions = env.action_space
    n_features = env.observation_space.shape
    q_list = []
    net = DQN(learning_rate, gamma, epsilon, n_actions, n_features)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    Episode = 100
    for episode in range(Episode):
        q_value = 0
        s = env.reset()
        # print(env.action_space.sample())
        while True:
            env.render()
            
            a = net.choose_action(np.expand_dims(s, 0)).numpy()
            # print(a)
            a = a[0]
            s_, r, done, info= env.step(a)
            
            # y_pred = net(np.expand_dims(s, 0)) * tf.one_hot(a, depth=2)
            # todo the shape is wrong
            q = net(np.expand_dims(s_, 0))
            y_true = r + gamma * tf.reduce_max(q)
            # print(f'[INFO]    y_pred:{y_pred} \n y_true:{y_true} ')
            with tf.GradientTape() as tape:
                loss = losses.mean_squared_error(
                    y_true=y_true,
                    y_pred=net.call(np.expand_dims(s, 0)) * tf.one_hot(a, depth=2)
                    )
                # loss = losses.mean_absolute_error(y_pred=y_pred, y_true=y_true)
                # loss = tf.reduce_mean(y_pred)
                # print(loss)

    
            grad = tape.gradient(loss, net.variables)
            # print(grad)
            optimizer.apply_gradients(grads_and_vars= zip(grad, net.variables) )
            # print(q)
            s = s_
            if done:
                q_value = q
                break
        q_list.append(q_value)
    # print(q_list)
            
            
