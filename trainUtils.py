"""
Example script for training an agent in the Auction Environment
using Q-Learning 
"""

import numpy as np
import matplotlib.pyplot as plt
from auctionEnv import AuctionEnv


class QLearningAgent:
    
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        
        Args:
            n_actions: Number of possible actions
            learning_rate: alpha
            discount_factor: gamma
            epsilon
            epsilon_decay
            epsilon_min
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        #Q-Table - we use a dictionary for discretized states
        self.q_table = {}
    
    def discretize_state(self, state):
        """
        Discretize the continuous state to use a Q-table.   
        We use feature engineering too.
        """
        round_num = int(state[0])
        
        #discretize capitals in ranges of $5 
        if state[1] < 30:
            agent_capital = int(state[1] / 2) * 2  
        elif state[1] < 60:
            agent_capital = int(state[1] / 5) * 5  
        else:
            agent_capital = int(state[1] / 10) * 10  
                
        opponent_capital = int(state[2] / 10) * 10
        
        item_value = int(state[3] / 5) * 5
        
        #FEATURE ENGINEERING, more informative than absolut values
        score_difference = state[4] - state[5]
        discretized_score_diff = int(score_difference / 20) * 20 
        
        #more little state
        return (round_num, agent_capital, opponent_capital, 
                item_value, discretized_score_diff)
    
    def get_q_value(self, state, action):
        """
        Args:
            state: discreticed state
            action
        """
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state, training=True):
        """
        Choose an action using epsilon-greedy policy.
        Args:
            state: actual state
            training: if it is in training
        
        Returns:
            int: chosen action
        """
        discrete_state = self.discretize_state(state)
        
        #exploration vs exploitation
        if training and np.random.random() < self.epsilon:
            #random action
            return np.random.randint(0, self.n_actions)
        else:
            #best action according to Q-table
            q_values = [self.get_q_value(discrete_state, a) 
                       for a in range(self.n_actions)]
            max_q = max(q_values)
            #if draw
            best_actions = [a for a in range(self.n_actions) 
                          if q_values[a] == max_q]
            return np.random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, done):

        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        current_q = self.get_q_value(discrete_state, action)
        
        #maximum Q value of the next state
        if done:
            max_next_q = 0
        else:
            next_q_values = [self.get_q_value(discrete_next_state, a) 
                           for a in range(self.n_actions)]
            max_next_q = max(next_q_values)
        
        #q-learning ecuation
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        #update q table
        self.q_table[(discrete_state, action)] = new_q
    
    def decay_epsilon(self):
        #reduce epsilon to decrease exploration.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def evaluate_agent(agent, n_episodes=100, render=False):
    """
    Args:
        agent
        n_episodes
        render: if render the episodes
    
    Returns:
        dict: evaluation metrics
    """
    env = AuctionEnv(initial_capital=100, render_mode='human' if render else None)
    
    rewards = []
    wins = []
    
    print(f"\nevaluating agent during  {n_episodes} apisodes...")
    
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        if render and episode < 3:  #only render first 3
            print(f"\n{'='*70}")
            print(f"Episode: {episode + 1}")
            print(f"{'='*70}")
            env.render()
        
        while not done:
            action = agent.choose_action(state, training=False)
            
            if render and episode < 3:
                action_pct = action * 10
                print(f"\n>>> Agent decides to bid: {action_pct}% of the item value")
                input("Press Enter to continue...")
            
            next_state, reward, done, truncated, info = env.step(action)
            
            if render and episode < 3:
                env.render()
            
            total_reward += reward
            state = next_state
        
        rewards.append(total_reward)
        wins.append(1 if info['is_winning'] else 0)
        
        if render and episode < 3:
            print(f"\n{'='*70}")
            print(f"EPISODE RESULT {episode + 1}")
            print(f"{'='*70}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"¿Agent wins? {'YES' if info['is_winning'] else 'NO'}")
            print(f"Final agent value: ${info['agent_total_value']:.2f}")
            print(f"Final oponent value: ${info['opponent_total_value']:.2f}")
    
    metrics = {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'win_rate': np.mean(wins) * 100,
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }
    
    print(f"\n{'='*70}")
    print("EVALUATION METRICS")
    print(f"{'='*70}")
    print(f"Average reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Min reward: {metrics['min_reward']:.2f}")
    print(f"Max reward: {metrics['max_reward']:.2f}")
    
    return metrics

