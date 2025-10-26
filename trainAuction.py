"""
Script de ejemplo para entrenar un agente en el Environment de Subastas
usando Q-Learning (método tabular de RL)
"""

import numpy as np
import matplotlib.pyplot as plt
from auctionEnv import AuctionEnv


class QLearningAgent:
    """
    Agente que usa Q-Learning para aprender a jugar subastas.
    """
    
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.95, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            n_actions: Número de acciones posibles
            learning_rate: Tasa de aprendizaje (alpha)
            discount_factor: Factor de descuento (gamma)
            epsilon: Probabilidad inicial de exploración
            epsilon_decay: Decaimiento de epsilon
            epsilon_min: Epsilon mínimo
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Tabla Q - usamos un diccionario para estados discretizados
        self.q_table = {}
    
    def discretize_state(self, state):
        """
        Discretiza el estado continuo para usar tabla Q.
        
        MEJORADO: Discretización más fina y feature engineering
        """
        round_num = int(state[0])
        
        # Discretizar capitales en rangos de $5 (más preciso)
        if state[1] < 30:
            agent_capital = int(state[1] / 2) * 2  # Muy preciso cuando queda poco
        elif state[1] < 60:
            agent_capital = int(state[1] / 5) * 5  # Preciso en rango medio
        else:
            agent_capital = int(state[1] / 10) * 10  # Menos preciso con mucho capital
                
        # El capital del oponente es menos crítico, $10 está bien
        opponent_capital = int(state[2] / 10) * 10
        
        # Discretizar valor del item en rangos de $5 (crucial)
        item_value = int(state[3] / 5) * 5
        
        # FEATURE ENGINEERING: Usar la *diferencia* de puntuación
        # Es más informativo que los valores absolutos
        score_difference = state[4] - state[5]
        discretized_score_diff = int(score_difference / 20) * 20 # Bloques de 20
        
        # Estado más pequeño y más informativo
        return (round_num, agent_capital, opponent_capital, 
                item_value, discretized_score_diff)
    
    def get_q_value(self, state, action):
        """
        Obtiene el valor Q para un par estado-acción.
        
        Args:
            state: Estado discretizado
            action: Acción
        
        Returns:
            float: Valor Q
        """
        return self.q_table.get((state, action), 0.0)
    
    def choose_action(self, state, training=True):
        """
        Elige una acción usando política epsilon-greedy.
        
        Args:
            state: Estado actual
            training: Si está en modo entrenamiento
        
        Returns:
            int: Acción elegida
        """
        discrete_state = self.discretize_state(state)
        
        # Exploración vs Explotación
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.randint(0, self.n_actions)
        else:
            # Explotación: mejor acción según Q-table
            q_values = [self.get_q_value(discrete_state, a) 
                       for a in range(self.n_actions)]
            max_q = max(q_values)
            # Si hay empate, elegir aleatoriamente entre las mejores
            best_actions = [a for a in range(self.n_actions) 
                          if q_values[a] == max_q]
            return np.random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Actualiza el valor Q usando la ecuación de Q-Learning.
        
        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa recibida
            next_state: Siguiente estado
            done: Si el episodio terminó
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Valor Q actual
        current_q = self.get_q_value(discrete_state, action)
        
        # Valor Q máximo del siguiente estado
        if done:
            max_next_q = 0
        else:
            next_q_values = [self.get_q_value(discrete_next_state, a) 
                           for a in range(self.n_actions)]
            max_next_q = max(next_q_values)
        
        # Ecuación de Q-Learning
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        
        # Actualizar tabla Q
        self.q_table[(discrete_state, action)] = new_q
    
    def decay_epsilon(self):
        """Reduce epsilon para disminuir exploración."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(n_episodes=1000, render_frequency=100):
    """
    Entrena el agente Q-Learning en el environment de subastas.
    
    Args:
        n_episodes: Número de episodios de entrenamiento
        render_frequency: Cada cuántos episodios renderizar
    
    Returns:
        agent: Agente entrenado
        rewards_history: Historial de recompensas
        wins_history: Historial de victorias
    """
    # Crear environment
    env = AuctionEnv(initial_capital=100)
    
    # Crear agente
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Historial
    rewards_history = []
    wins_history = []
    
    print(f"Entrenando agente durante {n_episodes} episodios...")
    print(f"Acciones disponibles: {env.action_space.n} (pujar 0%-100% en pasos de 10%)")
    
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        # Renderizar algunos episodios
        if (episode + 1) % render_frequency == 0:
            print(f"\n{'='*60}")
            print(f"Episodio {episode + 1}/{n_episodes}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Estados en Q-table: {len(agent.q_table)}")
        
        while not done:
            # Elegir acción
            action = agent.choose_action(state, training=True)
            
            # Ejecutar acción
            next_state, reward, done, truncated, info = env.step(action)
            
            # Actualizar Q-value
            agent.update_q_value(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
        
        # Guardar métricas
        rewards_history.append(total_reward)
        wins_history.append(1 if info['is_winning'] else 0)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Imprimir progreso
        if (episode + 1) % render_frequency == 0:
            avg_reward = np.mean(rewards_history[-render_frequency:])
            win_rate = np.mean(wins_history[-render_frequency:]) * 100
            print(f"Recompensa promedio (últimos {render_frequency}): {avg_reward:.2f}")
            print(f"Tasa de victoria (últimos {render_frequency}): {win_rate:.1f}%")
    
    print("\n¡Entrenamiento completado!")
    return agent, rewards_history, wins_history


def evaluate_agent(agent, n_episodes=100, render=False):
    """
    Evalúa el agente entrenado.
    
    Args:
        agent: Agente a evaluar
        n_episodes: Número de episodios de evaluación
        render: Si renderizar los episodios
    
    Returns:
        dict: Métricas de evaluación
    """
    env = AuctionEnv(initial_capital=100, render_mode='human' if render else None)
    
    rewards = []
    wins = []
    
    print(f"\nEvaluando agente durante {n_episodes} episodios...")
    
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        if render and episode < 3:  # Solo renderizar los primeros 3
            print(f"\n{'='*70}")
            print(f"EPISODIO DE EVALUACIÓN {episode + 1}")
            print(f"{'='*70}")
            env.render()
        
        while not done:
            action = agent.choose_action(state, training=False)
            
            if render and episode < 3:
                action_pct = action * 10
                print(f"\n>>> Agente decide pujar: {action_pct}% del capital")
                input("Presiona Enter para continuar...")
            
            next_state, reward, done, truncated, info = env.step(action)
            
            if render and episode < 3:
                env.render()
            
            total_reward += reward
            state = next_state
        
        rewards.append(total_reward)
        wins.append(1 if info['is_winning'] else 0)
        
        if render and episode < 3:
            print(f"\n{'='*70}")
            print(f"RESULTADO EPISODIO {episode + 1}")
            print(f"{'='*70}")
            print(f"Recompensa total: {total_reward:.2f}")
            print(f"¿Ganó? {'SÍ' if info['is_winning'] else 'NO'}")
            print(f"Valor final agente: ${info['agent_total_value']:.2f}")
            print(f"Valor final oponente: ${info['opponent_total_value']:.2f}")
    
    # Calcular métricas
    metrics = {
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'win_rate': np.mean(wins) * 100,
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards)
    }
    
    print(f"\n{'='*70}")
    print("MÉTRICAS DE EVALUACIÓN")
    print(f"{'='*70}")
    print(f"Recompensa promedio: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Tasa de victoria: {metrics['win_rate']:.1f}%")
    print(f"Recompensa mínima: {metrics['min_reward']:.2f}")
    print(f"Recompensa máxima: {metrics['max_reward']:.2f}")
    
    return metrics


def plot_training_results(rewards_history, wins_history, window=50):
    """
    Grafica los resultados del entrenamiento.
    
    Args:
        rewards_history: Historial de recompensas
        wins_history: Historial de victorias
        window: Ventana para promedio móvil
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Recompensas
    ax1.plot(rewards_history, alpha=0.3, label='Recompensa por episodio')
    
    # Promedio móvil de recompensas
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, 
                                np.ones(window)/window, 
                                mode='valid')
        ax1.plot(range(window-1, len(rewards_history)), 
                moving_avg, 
                label=f'Promedio móvil ({window} episodios)',
                linewidth=2)
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Recompensas durante el Entrenamiento')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tasa de victoria
    if len(wins_history) >= window:
        win_rate = np.convolve(wins_history, 
                              np.ones(window)/window, 
                              mode='valid') * 100
        ax2.plot(range(window-1, len(wins_history)), win_rate, linewidth=2)
    
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Tasa de Victoria (%)')
    ax2.set_title(f'Tasa de Victoria (promedio móvil {window} episodios)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/training_results.png', dpi=150, bbox_inches='tight')
    print("\nGráfica guardada en: training_results.png")
    plt.close()


def main():
    """Función principal."""
    print("="*70)
    print("ENTRENAMIENTO DE AGENTE RL PARA SUBASTAS")
    print("="*70)
    
    # Entrenar agente
    agent, rewards_history, wins_history = train_agent(
        n_episodes=2000,
        render_frequency=200
    )
    
    # Graficar resultados
    plot_training_results(rewards_history, wins_history)
    
    # Evaluar agente
    metrics = evaluate_agent(agent, n_episodes=100, render=True)
    
    print("\n¡Proceso completado!")
    print(f"Tamaño de Q-table: {len(agent.q_table)} entradas")


if __name__ == "__main__":
    main()