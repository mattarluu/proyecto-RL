"""
ENTRENAMIENTO MEJORADO: Contra Múltiples Oponentes
El agente entrena rotando entre diferentes estrategias de oponentes
"""

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import matplotlib.pyplot as plt
from auctionEnv import AuctionEnv
from trainAuction import QLearningAgent


def train_agent_multi_opponent(n_episodes=2000, 
                                rotate_strategy=True,
                                render_frequency=200):
    """
    Entrena al agente contra MÚLTIPLES estrategias de oponentes.
    
    Args:
        n_episodes: Número total de episodios
        rotate_strategy: Si rotar entre estrategias cada N episodios
        render_frequency: Cada cuántos episodios mostrar progreso
    
    Returns:
        agent: Agente entrenado
        rewards_history: Historial de recompensas
        wins_history: Historial de victorias
    """
def train_agent_multi_opponent(n_episodes=2000, 
                                rotate_strategy=True,
                                render_frequency=200):
    """
    Entrena al agente contra MÚLTIPLES estrategias de oponentes.
    
    MEJORADO: Ahora con epsilon decay adaptativo según número de episodios
    
    Args:
        n_episodes: Número total de episodios
        rotate_strategy: Si rotar entre estrategias cada N episodios
        render_frequency: Cada cuántos episodios mostrar progreso
    
    Returns:
        agent: Agente entrenado
        rewards_history: Historial de recompensas
        wins_history: Historial de victorias
        strategy_history: Historial de estrategias usadas
    """
    env = AuctionEnv(initial_capital=100)
    
    # EPSILON DECAY ADAPTATIVO según número de episodios
    # Para que epsilon llegue a ~0.05 al final
    target_epsilon = 0.05
    epsilon_decay = np.exp(np.log(target_epsilon) / n_episodes)
    
    print(f"📊 Epsilon decay calculado: {epsilon_decay:.6f} (para {n_episodes} episodios)")
    
    # HIPERPARÁMETROS MEJORADOS
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.5,            # Más alto para aprender más rápido
        discount_factor=0.85,
        epsilon=0.5,
        epsilon_decay=epsilon_decay,  # ADAPTATIVO al número de episodios
        epsilon_min=0.05              # Mínimo más alto para seguir explorando
    )
    
    # Estrategias disponibles
    strategies = ['random', 'fixed',  'adaptive','fixed', 'aggressive', 'fixed']
    
    # Historial
    rewards_history = []
    wins_history = []
    strategy_history = []
    
    print("="*70)
    print("ENTRENAMIENTO MEJORADO: Contra Múltiples Oponentes")
    print("="*70)
    print(f"Total de episodios: {n_episodes}")
    print(f"Estrategias: {strategies}")
    print(f"Rotación: {'Sí' if rotate_strategy else 'No'}")
    print()
    
    episodes_per_strategy = n_episodes // len(strategies)
    current_strategy_idx = 0
    
    # Contador de victorias por ventana para calcular win rate
    recent_wins = []
    window_size = 200
    
    for episode in range(n_episodes):
        # Rotar estrategia cada N episodios
        if rotate_strategy and episode % episodes_per_strategy == 0:
            strategy = strategies[current_strategy_idx % len(strategies)]
            env.set_opponent_strategy(strategy)
            if episode > 0:
                print(f"\n🔄 Cambiando a estrategia: '{strategy}'")
            current_strategy_idx += 1
        
        state, info = env.reset()
        total_reward = 0
        done = False
        
        # Jugar episodio
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
        
        # Registrar métricas
        rewards_history.append(total_reward)
        is_win = 1 if info['is_winning'] else 0
        wins_history.append(is_win)
        recent_wins.append(is_win)
        
        # Mantener ventana de victorias recientes
        if len(recent_wins) > window_size:
            recent_wins.pop(0)
        
        strategy_history.append(strategy)
        agent.decay_epsilon()
        
        # Imprimir progreso
        if (episode + 1) % render_frequency == 0:
            avg_reward = np.mean(rewards_history[-window_size:])
            win_rate = np.mean(recent_wins) * 100
            print(f"Episodio {episode + 1}/{n_episodes} | "
                  f"Estrategia: {strategy:10s} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Win%: {win_rate:5.1f}%")
    
    print("\n" + "="*70)
    print("✅ ¡Entrenamiento completado!")
    print("="*70)
    print(f"Estados explorados: {len(agent.q_table)}")
    print(f"Epsilon final: {agent.epsilon:.4f}")
    print(f"Win rate promedio: {np.mean(wins_history[-window_size:])*100:.1f}%")
    print(f"Reward promedio: {np.mean(rewards_history[-window_size:]):.2f}")
        
    # Guardar métricas
    rewards_history.append(total_reward)
    wins_history.append(1 if info['is_winning'] else 0)
    strategy_history.append(env.opponent_strategy)
    
    # Decay epsilon
    agent.decay_epsilon()
    
    # Progreso
    if (episode + 1) % render_frequency == 0:
        avg_reward = np.mean(rewards_history[-render_frequency:])
        win_rate = np.mean(wins_history[-render_frequency:]) * 100
        print(f"Episodio {episode + 1}/{n_episodes} | "
                f"Estrategia: {env.opponent_strategy:<10} | "
                f"Epsilon: {agent.epsilon:.4f} | "
                f"Reward: {avg_reward:>6.2f} | "
                f"Win%: {win_rate:>5.1f}%")
    
    print("\n✅ ¡Entrenamiento completado!")
    print(f"Tamaño Q-table: {len(agent.q_table)} estados")
    
    return agent, rewards_history, wins_history, strategy_history


def train_agent_curriculum(n_episodes_per_stage=500):
    """
    Entrenamiento con CURRICULUM LEARNING:
    Empieza con oponentes fáciles y aumenta dificultad.
    
    MEJORADO: Epsilon decay adaptativo según número total de episodios
    
    Esta es una técnica común en RL para mejorar el aprendizaje.
    """
    env = AuctionEnv(initial_capital=100)
    
    # Calcular episodios totales
    n_total_episodes = n_episodes_per_stage * 6 # 4 etapas
    
    # EPSILON DECAY ADAPTATIVO
    target_epsilon = 0.05
    epsilon_decay = np.exp(np.log(target_epsilon) / n_total_episodes)
    
    print(f"📊 Total episodios: {n_total_episodes}")
    print(f"📊 Epsilon decay: {epsilon_decay:.6f}")
    
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.5,           # Más alto para aprender rápido
        discount_factor=0.85,
        epsilon=0.5,
        epsilon_decay=epsilon_decay, # ADAPTATIVO
        epsilon_min=0.05
    )
    
    # Curriculum: de fácil a difícil
    curriculum = [
        ('random', 'Aleatorio'),
        ('fixed', 'Fijo '),
        ('aggressive', 'Agresivo '),
        ('fixed', 'Fijo '),
        ('adaptive', 'Adaptativo '),
        ('fixed', 'Fijo ')
        
    ]
    
    print("="*70)
    print("ENTRENAMIENTO CON CURRICULUM LEARNING")
    print("="*70)
    print("Estrategia: Empezar fácil, aumentar dificultad gradualmente")
    print()
    
    all_rewards = []
    all_wins = []
    
    for stage, (strategy, description) in enumerate(curriculum, 1):
        print(f"\n📚 ETAPA {stage}/6: {description}")
        print(f"   Entrenando {n_episodes_per_stage} episodios contra '{strategy}'")
        
        env.set_opponent_strategy(strategy)
        stage_rewards = []
        stage_wins = []
        
        for episode in range(n_episodes_per_stage):
            state, info = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.choose_action(state, training=True)
                next_state, reward, done, truncated, info = env.step(action)
                agent.update_q_value(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            
            stage_rewards.append(total_reward)
            stage_wins.append(1 if info['is_winning'] else 0)
            agent.decay_epsilon()
            
            # Progreso cada 100 episodios
            if (episode + 1) % 100 == 0:
                recent_wr = np.mean(stage_wins[-100:]) * 100
                print(f"   Episodio {episode + 1}/{n_episodes_per_stage} | "
                      f"Metodo: {strategy} | "
                      f"Win Rate: {recent_wr:.1f}%")
        
        all_rewards.extend(stage_rewards)
        all_wins.extend(stage_wins)
        
        final_wr = np.mean(stage_wins[-100:]) * 100
        print(f"   ✓ Etapa completada. Win Rate final: {final_wr:.1f}%")
    
    print("\n✅ ¡Curriculum completado!")
    print(f"Tamaño Q-table: {len(agent.q_table)} estados")
    
    return agent, all_rewards, all_wins


def evaluate_against_all(agent, n_matches=50):
    """Evalúa un agente contra todas las estrategias."""
    env = AuctionEnv(initial_capital=100)
    strategies = ['random', 'fixed', 'adaptive', 'aggressive']
    
    results = {'by_strategy': {}}
    total_wins = 0
    
    for strategy in strategies:
        env.set_opponent_strategy(strategy)
        wins = 0
        
        for _ in range(n_matches):
            state, _ = env.reset()
            done = False
            
            while not done:
                action = agent.choose_action(state, training=False)
                state, _, done, _, info = env.step(action)
            
            if info['is_winning']:
                wins += 1
        
        wr = (wins / n_matches) * 100
        results['by_strategy'][strategy] = wr
        total_wins += wins
    
    results['overall_wr'] = (total_wins / (n_matches * len(strategies))) * 100
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenamiento Mejorado de Agente RL')
    parser.add_argument('--metodo', type=str, default='multi',
                       choices=['multi', 'curriculum', 'adaptive', 'compare'],
                       help='Método de entrenamiento')
    parser.add_argument('--episodios', type=int, default=2000,
                       help='Número de episodios')
    
    args = parser.parse_args()
    
    if args.metodo == 'multi':
        print("🚀 Entrenamiento Multi-Oponente")
        agent, rewards, wins, strategies = train_agent_multi_opponent(
            n_episodes=args.episodios
        )
        
    elif args.metodo == 'curriculum':
        print("📚 Entrenamiento con Curriculum Learning")
        agent, rewards, wins = train_agent_curriculum(
            n_episodes_per_stage=args.episodios // 4
        )
    
    # Evaluar agente final
    print("\n" + "="*70)
    print("EVALUACIÓN FINAL")
    print("="*70)
    results = evaluate_against_all(agent, n_matches=100)
    
    print(f"\nWin Rate General: {results['overall_wr']:.1f}%\n")
    for strategy, wr in results['by_strategy'].items():
        print(f"  vs {strategy:<12}: {wr:>5.1f}%")
    
    print("\n✅ Entrenamiento y evaluación completados!")