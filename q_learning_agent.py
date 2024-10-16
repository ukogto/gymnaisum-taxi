import gymnasium as gym
import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95
    ):
        """Initialize a Q-learning agent with an empty dictionary of state-action values (q_values).
        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q values
        """
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor 

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Initialize an empty dictionary of state-action values
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs) -> int:
        """An epsilon-greedy action selection. Returns the best action with probability (1 - epsilon),
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # obs: observation is the state of env (combination of taxi pos, human pos, and all the stops)

        if np.random.uniform() < self.epsilon :
            return self.env.action_space.sample()

        # rest (1-epsilon) times here:
        current_best_action = np.argmax(self.q_values[obs])
        return np.argmax(self.q_values[obs])

    def update(self, obs, action, reward, terminated, next_obs):
        """Update the Q-value of the chosen action.
        Args:
            obs: The current observation
            action: The action chosen by the agent
            reward: The reward received for taking that action
            terminated: Whether the episode has terminated after taking that action
            next_obs: The observation received from the environment after taking that action
        """
        # max_a' Q(s', a') is the current approximation of what the max reward learnt from prev iteration
        # and  Q(s, a) is the prev state's reward
        # calculate termporal difference times learning rate and add it to Q value
        future_q_value_basedon_best_possible_action = 0
        if not terminated: # if s' is final then Q(s', a') is zero
            future_q_value_basedon_best_possible_action = np.max(self.q_values[next_obs]) 

        self.q_values[obs][action] += \
            self.learning_rate*(
                reward
                + self.discount_factor*future_q_value_basedon_best_possible_action
                - self.q_values[obs][action])

    def decay_epsilon(self):
        """Decrease the exploration rate epsilon until it reaches its final value"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)