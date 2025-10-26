"""
The main training fuction.
The agent has two modes of training:
- AGainst multi opponents cyclically
- Against multi opponents by curriculum learning, used most of the time
"""

import sys
sys.path.insert(0, '/mnt/user-data/uploads')

import numpy as np
import matplotlib.pyplot as plt
from auctionEnv import AuctionEnv
from trainUtils import QLearningAgent

def train_agent_multi_opponent(n_episodes=2000, 
                               rotate_strategy=True,
                               render_frequency=200,
                               episodes_per_rotation=None):  
    """
    Trains the agent against multiple opponent strategies cyclically.

    Args:
        n_episodes: Total number.
        rotate_strategy: Whether to rotate between strategies.
        render_frequency: How many episodes every render progress.
        episodes_per_rotation: Number of episodes each strategy lasts before rotating.

    Returns:
        agent: Trained agent
        rewards_history: Rewards history
        wins_history: Wins history
        strategy_history: History of strategies used
    """
    env = AuctionEnv(initial_capital=100)
    
    #according to number of episodes
    target_epsilon = 0.05
    epsilon_decay = np.exp(np.log(target_epsilon) / n_episodes)

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.5,
        discount_factor=0.85,
        epsilon=0.5,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.05
    )
    
    strategies = ['random', 'fixed', 'aggressive','fixed', 'adaptive', 'fixed']
    
    #history -- informastive
    rewards_history = []
    wins_history = []
    strategy_history = []
    
    if episodes_per_rotation is None:
        episodes_per_rotation = n_episodes // len(strategies)
    
    #avoid errors
    episodes_per_rotation = max(1, episodes_per_rotation)
    
    current_strategy_idx = 0
    
    print("="*70)
    print("Against multiple opponents")
    print("="*70)
    print(f"Total episodes: {n_episodes}")
    print(f"Strategies: {strategies}")
    print(f"rotation: {'YES' if rotate_strategy else 'NO'} (every {episodes_per_rotation} episodes)") 
    print(f"Epsilon decay calculado: {epsilon_decay:.6f}")
    print()
    
    recent_wins = []
    window_size = 200
    
    for episode in range(n_episodes):
        if rotate_strategy and episode % episodes_per_rotation == 0:
            strategy = strategies[current_strategy_idx % len(strategies)]
            env.set_opponent_strategy(strategy)
            if episode > 0:
                print(f"\nChanging to: '{strategy}'")
            current_strategy_idx += 1
        
        state, info = env.reset()
        total_reward = 0
        done = False
        
        #play
        while not done:
            action = agent.choose_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
        
        rewards_history.append(total_reward)
        is_win = 1 if info['is_winning'] else 0
        wins_history.append(is_win)
        recent_wins.append(is_win)
        
        if len(recent_wins) > window_size:
            recent_wins.pop(0)
        
        strategy_history.append(strategy)
        agent.decay_epsilon()
        
        #progress
        if (episode + 1) % render_frequency == 0:
            avg_reward = np.mean(rewards_history[-window_size:])
            win_rate = np.mean(recent_wins) * 100
            
            current_strategy = env.opponent_strategy 
            print(f"Agent {episode + 1}/{n_episodes} | "
                  f"Strategy: {current_strategy:10s} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Reward: {avg_reward:6.2f} | "
                  f"Win%: {win_rate:5.1f}%")
    
    
    print("\n" + "="*70)
    print("¡Training completed!")
    print("="*70)
    print(f"Explored States: {len(agent.q_table)}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print(f"Average Win rate: {np.mean(wins_history[-window_size:])*100:.1f}%")
    print(f"Average Reward: {np.mean(rewards_history[-window_size:]):.2f}")
    
    rewards_history.append(total_reward)
    wins_history.append(1 if info['is_winning'] else 0)
    strategy_history.append(env.opponent_strategy)
    
    #plot_multi_opponent_results(rewards_history, wins_history, strategy_history)
    return agent, rewards_history, wins_history, strategy_history


def train_agent_curriculum(n_episodes_per_stage=500):
    """
    Training with CURRICULUM LEARNING:
    Start with easy opponents and increase the difficulty. 
    This order of opponents is due to the fact that it is more difficult against tome opponents.

    """
    env = AuctionEnv(initial_capital=100)
    
    #total episodes
    n_total_episodes = n_episodes_per_stage * 6 # 
    
    target_epsilon = 0.05
    epsilon_decay = np.exp(np.log(target_epsilon) / n_total_episodes)
    
    print(f"📊 Total episodios: {n_total_episodes}")
    print(f"📊 Epsilon decay: {epsilon_decay:.6f}")
    
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=0.5,           
        discount_factor=0.85,
        epsilon=0.5,
        epsilon_decay=epsilon_decay, 
        epsilon_min=0.05
    )
    
    #from easy to difficult
    curriculum = [
    ('random', 'Aleatorio'),
    ('fixed', 'Fijo'),
    ('aggressive', 'Agresivo'),
    ('fixed', 'Fijo'),
    ('adaptive', 'Adaptativo'),
    ('fixed', 'Fijo')
]
    
    print("="*70)
    print("TRAINING WITH CURRICULUM LEARNING")
    print("="*70)
    print("sTRATEGY: Start easy, gradually increase difficulty")
    print()
    
    all_rewards = []
    all_wins = []
    
    for stage, (strategy, description) in enumerate(curriculum, 1):
        print(f"\STAGE {stage}/{len(curriculum)}: {description}")
        print(f"   Training {n_episodes_per_stage} epìsodes against '{strategy}'")
        
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
            
            #progress each 100 episodes
            if (episode + 1) % 100 == 0:
                recent_wr = np.mean(stage_wins[-100:]) * 100
                print(f"   Episode {episode + 1}/{n_episodes_per_stage} | "
                      f"Metodo: {strategy} | "
                      f"Epsilon: {agent.epsilon:.4f} | "
                      f"Win Rate: {recent_wr:.1f}%")
        
        all_rewards.extend(stage_rewards)
        all_wins.extend(stage_wins)
        
        final_wr = np.mean(stage_wins[-100:]) * 100
        print(f"Stage completed. Final Win Rate: {final_wr:.1f}%")
    
    print("\n¡Curriculum completed!")
    print(f"Q-table shape: {len(agent.q_table)} states")
    curriculum_stages = [
            ('random', 'Random'),
            ('fixed', 'Fixed'),
            ('aggressive', 'Aggressive'),
            ('fixed', 'Fixed'),
            ('adaptive', 'Adaptive'),
            ('fixed', 'Fixed')
        ]
    plot_curriculum_results(all_rewards, all_wins, curriculum_stages)
    
    
    return agent, all_rewards, all_wins


def evaluate_against_all(agent, n_matches=50):
    """Evaluates an agent against all strategies"""
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

def plot_curriculum_results(all_rewards, all_wins, curriculum_stages, window=50):
    """
    Visualize training results with Curriculum Learning.
    """
    n_stages = len(curriculum_stages)
    episodes_per_stage = len(all_rewards) // n_stages
    
    #plot 1
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    episodes = range(len(all_rewards))
    
    if len(all_rewards) >= window:
        moving_avg_rewards = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(all_rewards)), moving_avg_rewards, 
                linewidth=2, color='blue', label=f'Average ({window} episodes)')
    

    ax1.plot(episodes, all_rewards, alpha=0.3, color='lightblue', label='Episode reward')
    
    stage_boundaries = [i * episodes_per_stage for i in range(n_stages + 1)]
    for i, boundary in enumerate(stage_boundaries[:-1]):
        ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.7, 
                   label='Cambio de etapa' if i == 0 else "")
        
        mid_point = boundary + episodes_per_stage // 2
        if mid_point < len(all_rewards):
            strategy, description = curriculum_stages[i]
            ax1.text(mid_point, ax1.get_ylim()[1] * 0.9, f'{description}', 
                    ha='center', va='top', fontsize=9, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Evolution of Rewards - Curriculum Learning', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    #plot 2
    if len(all_wins) >= window:
        moving_win_rate = np.convolve(all_wins, np.ones(window)/window, mode='valid') * 100
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i in range(n_stages):
            start_idx = i * episodes_per_stage
            end_idx = (i + 1) * episodes_per_stage
            start_plot = max(0, start_idx - window + 1)
            end_plot = min(len(moving_win_rate), end_idx - window + 1)
            
            if start_plot < end_plot:
                strategy, description = curriculum_stages[i]
                color = colors[i % len(colors)]
                
                ax2.plot(range(start_plot + window - 1, end_plot + window - 1), 
                        moving_win_rate[start_plot:end_plot], 
                        linewidth=3, color=color, label=f'{description}')
    
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% (balance)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate by Stage - Curriculum Learning', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    #plot 3
    stage_rewards = []
    stage_win_rates = []
    stage_labels = []
    
    for i in range(n_stages):
        start_idx = i * episodes_per_stage
        end_idx = (i + 1) * episodes_per_stage
        
        if end_idx <= len(all_rewards):
            stage_reward_data = all_rewards[start_idx:end_idx]
            stage_win_data = all_wins[start_idx:end_idx]
            
            stage_rewards.append(stage_reward_data)
            stage_win_rates.append(np.mean(stage_win_data) * 100)
            strategy, description = curriculum_stages[i]
            stage_labels.append(f'{description}\n({strategy})')
    
    box_plot = ax3.boxplot(stage_rewards, labels=stage_labels, patch_artist=True)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, box in enumerate(box_plot['boxes']):
        box.set(facecolor=colors[i % len(colors)], alpha=0.7)
    
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward Distribution by Stage', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    #plot 4
    final_win_rates = []
    for i in range(n_stages):
        start_idx = max(0, (i + 1) * episodes_per_stage - 100)
        end_idx = (i + 1) * episodes_per_stage
        
        if end_idx <= len(all_wins) and start_idx < end_idx:
            final_win_rate = np.mean(all_wins[start_idx:end_idx]) * 100
            final_win_rates.append(final_win_rate)
        else:
            final_win_rates.append(0)

    bars = ax4.bar(range(len(final_win_rates)), final_win_rates, 
                  color=colors[:len(final_win_rates)], alpha=0.7, edgecolor='black')
    
    ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Limit 50%')
    ax4.set_xlabel('Stage')
    ax4.set_ylabel('Final Win rate (%)')
    ax4.set_title('Final Win Rate by Stage (las 100 episodes)', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(final_win_rates)))
    ax4.set_xticklabels([stages[1] for stages in curriculum_stages[:len(final_win_rates)]], 
                       rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    for bar, wr in zip(bars, final_win_rates):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{wr:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/curriculum_learning_results.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
   
    print(f"\n{'='*70}")
    print("CURRICULUM LEARNING STATISTICS")
    print(f"{'='*70}")
    
    for i in range(n_stages):
        start_idx = i * episodes_per_stage
        end_idx = (i + 1) * episodes_per_stage
        
        if end_idx <= len(all_rewards):
            stage_rewards = all_rewards[start_idx:end_idx]
            stage_wins = all_wins[start_idx:end_idx]
            
            strategy, description = curriculum_stages[i]
            
            early_rewards = stage_rewards[:100] if len(stage_rewards) >= 100 else stage_rewards
            early_win_rate = np.mean(stage_wins[:100]) * 100 if len(stage_wins) >= 100 else np.mean(stage_wins) * 100
            

            late_rewards = stage_rewards[-100:] if len(stage_rewards) >= 100 else stage_rewards
            late_win_rate = np.mean(stage_wins[-100:]) * 100 if len(stage_wins) >= 100 else np.mean(stage_wins) * 100
            
            
            improvement = late_win_rate - early_win_rate
            
            print(f"\nStage {i+1}: {description} ({strategy})")
            print(f"   Episodes: {len(stage_rewards)}")
            print(f"   Win Rate: {early_win_rate:5.1f}% → {late_win_rate:5.1f}% " 
                  f"(Δ{improvement:+.1f}%)")
            print(f"   Average reward: {np.mean(stage_rewards):6.2f}")
    
    
    overall_win_rate = np.mean(all_wins) * 100
    final_win_rate = np.mean(all_wins[-100:]) * 100 if len(all_wins) >= 100 else overall_win_rate
    
    print(f"\n{'='*70}")
    print(f"GENERAL SUMMARY:")
    print(f"   Global Win Rate: {overall_win_rate:.1f}%")
    print(f"   Final Win Rate:  {final_win_rate:.1f}%")
    print(f"   Average reward: {np.mean(all_rewards):.2f}")
    print(f"   Total episodes: {len(all_rewards)}")
    print(f"{'='*70}")
    
    print("Curriculum Learning Charts saved in: plots/curriculum_learning_results.png")
    
def plot_multi_opponent_results(rewards_history, wins_history, strategy_history, window=100):
    """
    Visualiza los resultados del entrenamiento con Multi-Oponente.
    
    Args:
        rewards_history: Lista de todas las recompensas durante el entrenamiento
        wins_history: Lista de todas las victorias (1/0)
        strategy_history: Lista de estrategias usadas en cada episodio
        window: Tamaño de ventana para promedio móvil
    """
    if not rewards_history or not wins_history:
        print("❌ No hay datos suficientes para generar gráficas")
        return
    
    # Encontrar estrategias únicas y puntos de cambio
    unique_strategies = []
    strategy_changes = []
    current_strategy = strategy_history[0] if strategy_history else None
    
    for i, strategy in enumerate(strategy_history):
        if strategy != current_strategy:
            strategy_changes.append(i)
            current_strategy = strategy
            if strategy not in unique_strategies:
                unique_strategies.append(strategy)
    
    print(f"\n📊 Generando gráficas de Multi-Oponente...")
    print(f"   Estrategias únicas: {unique_strategies}")
    print(f"   Cambios de estrategia: {len(strategy_changes)}")
    print(f"   Total episodios: {len(rewards_history)}")
    
    # Crear figura con subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ===== GRÁFICA 1: Recompensas con cambios de estrategia =====
    episodes = range(len(rewards_history))
    
    # Calcular promedio móvil de recompensas
    if len(rewards_history) >= window:
        moving_avg_rewards = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards_history)), moving_avg_rewards, 
                linewidth=2, color='blue', label=f'Promedio móvil ({window} episodios)')
    
    # Recompensas individuales (transparentes)
    ax1.plot(episodes, rewards_history, alpha=0.3, color='lightblue', label='Recompensa por episodio')
    
    # Marcar cambios de estrategia
    for i, change_point in enumerate(strategy_changes):
        ax1.axvline(x=change_point, color='red', linestyle='--', alpha=0.7, 
                   label='Cambio de estrategia' if i == 0 else "")
        
        # Añadir etiqueta de estrategia
        if change_point < len(rewards_history):
            strategy = strategy_history[change_point]
            ax1.text(change_point, ax1.get_ylim()[1] * 0.9, f'{strategy}', 
                    ha='center', va='top', fontsize=8, rotation=45,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    ax1.set_xlabel('Episodio')
    ax1.set_ylabel('Recompensa')
    ax1.set_title('Evolución de Recompensas - Entrenamiento Multi-Oponente', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== GRÁFICA 2: Win Rate por estrategia =====
    if len(wins_history) >= window:
        # Calcular win rate móvil
        moving_win_rate = np.convolve(wins_history, np.ones(window)/window, mode='valid') * 100
        
        # Graficar win rate con colores por estrategia
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i in range(len(strategy_changes)):
            start_idx = strategy_changes[i]
            end_idx = strategy_changes[i+1] if i < len(strategy_changes)-1 else len(moving_win_rate) + window - 1
            
            # Ajustar índices para el promedio móvil
            start_plot = max(0, start_idx - window + 1)
            end_plot = min(len(moving_win_rate), end_idx - window + 1)
            
            if start_plot < end_plot and start_idx < len(strategy_history):
                strategy = strategy_history[start_idx]
                color = colors[i % len(colors)]
                
                ax2.plot(range(start_plot + window - 1, end_plot + window - 1), 
                        moving_win_rate[start_plot:end_plot], 
                        linewidth=3, color=color, label=f'{strategy}')
    
    ax2.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% (equilibrio)')
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('Tasa de Victoria (%)')
    ax2.set_title('Win Rate por Estrategia de Oponente', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # ===== GRÁFICA 3: Comparación de estrategias (Boxplot) =====
    strategy_rewards = {}
    strategy_win_rates = {}
    
    # Agrupar recompensas por estrategia
    for i, strategy in enumerate(strategy_history):
        if strategy not in strategy_rewards:
            strategy_rewards[strategy] = []
            strategy_win_rates[strategy] = []
        
        strategy_rewards[strategy].append(rewards_history[i])
        strategy_win_rates[strategy].append(wins_history[i])
    
    # Preparar datos para boxplot
    boxplot_data = []
    boxplot_labels = []
    for strategy in unique_strategies:
        if strategy in strategy_rewards and len(strategy_rewards[strategy]) > 0:
            boxplot_data.append(strategy_rewards[strategy])
            boxplot_labels.append(strategy)
    
    # Boxplot de recompensas por estrategia
    if boxplot_data:
        box_plot = ax3.boxplot(boxplot_data, labels=boxplot_labels, patch_artist=True)
        
        # Colorear los boxplots
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, box in enumerate(box_plot['boxes']):
            box.set(facecolor=colors[i % len(colors)], alpha=0.7)
    
    ax3.set_ylabel('Recompensa')
    ax3.set_title('Distribución de Recompensas por Estrategia', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # ===== GRÁFICA 4: Win Rate final por estrategia =====
    final_win_rates = []
    strategy_labels = []
    
    for strategy in unique_strategies:
        if strategy in strategy_win_rates and len(strategy_win_rates[strategy]) >= 100:
            # Calcular win rate de los últimos 100 episodios con esta estrategia
            win_data = strategy_win_rates[strategy]
            final_win_rate = np.mean(win_data[-100:]) * 100
            final_win_rates.append(final_win_rate)
            strategy_labels.append(strategy)
    
    # Gráfica de barras del win rate final por estrategia
    if final_win_rates:
        bars = ax4.bar(range(len(final_win_rates)), final_win_rates, 
                      color=colors[:len(final_win_rates)], alpha=0.7, edgecolor='black')
        
        ax4.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Límite 50%')
        ax4.set_xlabel('Estrategia')
        ax4.set_ylabel('Win Rate Final (%)')
        ax4.set_title('Win Rate Final por Estrategia (últimos 100 episodios)', fontsize=14, fontweight='bold')
        ax4.set_xticks(range(len(final_win_rates)))
        ax4.set_xticklabels(strategy_labels, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, wr in zip(bars, final_win_rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{wr:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Ajustar layout y guardar
    plt.tight_layout()
    plt.savefig('plots/multi_opponent_results.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # ===== IMPRIMIR ESTADÍSTICAS EN CONSOLA =====
    print(f"\n{'='*70}")
    print("📈 ESTADÍSTICAS DEL ENTRENAMIENTO MULTI-OPONENTE")
    print(f"{'='*70}")
    
    for strategy in unique_strategies:
        if strategy in strategy_rewards:
            rewards = strategy_rewards[strategy]
            wins = strategy_win_rates[strategy]
            
            if len(rewards) > 0:
                win_rate = np.mean(wins) * 100
                avg_reward = np.mean(rewards)
                
                print(f"\n🎯 Estrategia: {strategy}")
                print(f"   📊 Episodios: {len(rewards)}")
                print(f"   🏆 Win Rate: {win_rate:5.1f}%")
                print(f"   💰 Recompensa promedio: {avg_reward:6.2f}")
                
                # Calcular mejora si hay suficientes datos
                if len(wins) >= 200:
                    early_win_rate = np.mean(wins[:100]) * 100
                    late_win_rate = np.mean(wins[-100:]) * 100
                    improvement = late_win_rate - early_win_rate
                    print(f"   📈 Mejora: {early_win_rate:5.1f}% → {late_win_rate:5.1f}% (Δ{improvement:+.1f}%)")
    
    # Estadísticas generales
    overall_win_rate = np.mean(wins_history) * 100
    final_win_rate = np.mean(wins_history[-100:]) * 100 if len(wins_history) >= 100 else overall_win_rate
    
    print(f"\n{'='*70}")
    print(f"🎯 RESUMEN GENERAL:")
    print(f"   Win Rate Global: {overall_win_rate:.1f}%")
    print(f"   Win Rate Final:  {final_win_rate:.1f}%")
    print(f"   Recompensa Promedio: {np.mean(rewards_history):.2f}")
    print(f"   Total Episodios: {len(rewards_history)}")
    print(f"   Estrategias diferentes: {len(unique_strategies)}")
    print(f"{'='*70}")
    
    print("✅ Gráficas de Multi-Oponente guardadas en: plots/multi_opponent_results.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Agent Training')
    parser.add_argument('--method', type=str, default='multi',
                       choices=['multi', 'curriculum', 'adaptive', 'compare'],
                       help='Training method')
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Numbre of episodes')
    
    args = parser.parse_args()
    
    if args.metodo == 'multi':
        print("Multi-Oponente Training")
        agent, rewards, wins, strategies = train_agent_multi_opponent(
            n_episodes=args.episodios
        )
        
    elif args.metodo == 'curriculum':
        print("Curriculum Learning training")
        agent, rewards, wins = train_agent_curriculum(
            n_episodes_per_stage=args.episodios // 6
        )
    
    #final agent evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    results = evaluate_against_all(agent, n_matches=100)
    
    print(f"\nGeneral in Rate: {results['overall_wr']:.1f}%\n")
    for strategy, wr in results['by_strategy'].items():
        print(f"  vs {strategy:<12}: {wr:>5.1f}%")
    
    print("\nTraining and evaluation completed!")