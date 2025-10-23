import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import pickle


class QLearningAgent:
    """
    Agente que aprende usando Q-Learning (off-policy).
    
    Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    """
    
    def __init__(
        self,
        action_space_size: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            action_space_size: Número de acciones posibles
            learning_rate: Tasa de aprendizaje (alpha)
            discount_factor: Factor de descuento (gamma)
            epsilon_start: Epsilon inicial para epsilon-greedy
            epsilon_min: Epsilon mínimo
            epsilon_decay: Factor de decay del epsilon
        """
        self.action_space_size = action_space_size
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Tabla Q: usa defaultdict para inicializar en 0
        self.Q = defaultdict(lambda: np.zeros(action_space_size))
        
        # Estadísticas
        self.training_error = []
    
    def get_action(self, state: Tuple, training: bool = True) -> int:
        """
        Selecciona una acción usando política epsilon-greedy.
        
        Args:
            state: Estado actual (tupla)
            training: Si está en modo entrenamiento (usa epsilon)
        
        Returns:
            Acción seleccionada
        """
        state_key = tuple(state)
        
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(0, self.action_space_size)
        else:
            # Explotación: mejor acción según Q
            return np.argmax(self.Q[state_key])
    
    def update(
        self,
        state: Tuple,
        action: int,
        reward: float,
        next_state: Tuple,
        done: bool
    ):
        """
        Actualiza la tabla Q usando la regla de Q-Learning.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio terminó
        """
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # Valor actual de Q(s,a)
        current_q = self.Q[state_key][action]
        
        # Valor máximo de Q(s',a') (off-policy: usa max)
        if done:
            max_next_q = 0.0
        else:
            max_next_q = np.max(self.Q[next_state_key])
        
        # Target: r + γ·max(Q(s',a'))
        target = reward + self.gamma * max_next_q
        
        # Error TD
        td_error = target - current_q
        
        # Actualización: Q(s,a) ← Q(s,a) + α·[target - Q(s,a)]
        self.Q[state_key][action] = current_q + self.alpha * td_error
        
        # Guardar error para análisis
        self.training_error.append(abs(td_error))
    
    def decay_epsilon(self):
        """Decae epsilon después de cada episodio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """Guarda el agente en un archivo."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'Q': dict(self.Q),
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon
            }, f)
    
    def load(self, filepath: str):
        """Carga el agente desde un archivo."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.Q = defaultdict(lambda: np.zeros(self.action_space_size), data['Q'])
            self.alpha = data['alpha']
            self.gamma = data['gamma']
            self.epsilon = data['epsilon']

    def train_agent(
        env,
        agent,
        n_episodes: int = 5000,
        max_steps: int = 100,
        verbose: bool = True,
        save_path: str = None
        ) -> Dict[str, List]:
        """
        Entrena un agente en el entorno.

        Args:
            env: Entorno de Gymnasium
            agent: Agente de RL (Q-Learning, SARSA, etc.)
            n_episodes: Número de episodios de entrenamiento
            max_steps: Máximo de pasos por episodio
            verbose: Si mostrar progreso
            save_path: Ruta para guardar el agente

        Returns:
            Diccionario con métricas de entrenamiento
        """
        # Métricas
        episode_rewards = []
        episode_lengths = []
        epsilon_history = []
        win_rates = []
        capital_finals = []

        for episode in range(n_episodes):
            state, info = env.reset()
            total_reward = 0
            steps = 0
            done = False
    
            # Q-Learning o Expected SARSA
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Actualizar con (s, a, r, s')
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
                
            total_reward += reward
            steps += 1
            
            # Decaer epsilon
            agent.decay_epsilon()
            
            # Guardar métricas
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            epsilon_history.append(agent.epsilon)
            win_rates.append(info['agent_items_won'] / env.n_auctions)
            capital_finals.append(info['agent_capital'])
            
            # Mostrar progreso
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_win_rate = np.mean(win_rates[-100:])
                print(f"Episodio {episode + 1}/{n_episodes} | "
                        f"Reward: {avg_reward:.2f} | "
                        f"Win Rate: {avg_win_rate*100:.1f}% | "
                        f"Epsilon: {agent.epsilon:.3f}")

        # Guardar agente si se especifica ruta
        if save_path:
            agent.save(save_path)
            print(f"\n✓ Agente guardado en: {save_path}")

        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'epsilon_history': epsilon_history,
            'win_rates': win_rates,
            'capital_finals': capital_finals
        }
    
    def evaluate_agent(
        env,
        agent,
        n_episodes: int = 100,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evalúa un agente entrenado (sin exploración).
        
        Args:
            env: Entorno
            agent: Agente entrenado
            n_episodes: Número de episodios de evaluación
            render: Si renderizar
        
        Returns:
            Diccionario con estadísticas
        """
        rewards = []
        win_rates = []
        capitals = []
        
        for episode in range(n_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                # Sin exploración (training=False)
                action = agent.get_action(state, training=False)
                state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                
                if render and episode == 0:  # Solo renderizar primer episodio
                    env.render()
            
            rewards.append(total_reward)
            win_rates.append(info['agent_items_won'] / env.n_auctions)
            capitals.append(info['agent_capital'])
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_win_rate': np.mean(win_rates),
            'std_win_rate': np.std(win_rates),
            'mean_capital': np.mean(capitals),
            'std_capital': np.std(capitals)
        }