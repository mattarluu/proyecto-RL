def main():
    """Script principal para entrenar y comparar agentes."""
    
    from AuctionEnv import AuctionEnv
    from agent import QLearningAgent as a
    from utils import plot_training_results
    
    print("=" * 60)
    print("ENTRENAMIENTO DE AGENTES RL EN ENTORNO DE SUBASTAS")
    print("=" * 60)
    
    # Crear entorno
    env = AuctionEnv(n_auctions=10, initial_capital=100, opponent_strategy='smart')
    
    # Configuración de entrenamiento
    n_episodes = 5000
    
    # Hiperparámetros
    config = {
        'learning_rate': 0.1,
        'discount_factor': 0.95,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995
    }
    
    # ===== ENTRENAR Q-LEARNING =====
    print("\n" + "="*60)
    print("ENTRENANDO Q-LEARNING")
    print("="*60)
    
    agent_qlearning = a(
        action_space_size=env.action_space.n,
        **config
    )
    
    metrics_qlearning = a.train_agent(
        env=env,
        agent=agent_qlearning,
        n_episodes=n_episodes,
        save_path='qlearning_agent.pkl'
    )
    

    
    # ===== EVALUACIÓN =====
    print("\n" + "="*60)
    print("EVALUANDO AGENTES ENTRENADOS")
    print("="*60)
    
    results_qlearning = a.evaluate_agent(env, agent_qlearning, n_episodes=100)
    
    
    print("\nRESULTADOS (100 episodios de evaluación):")
    print("-" * 60)
    print(f"{'Algoritmo':<20} {'Reward':<15} {'Win Rate':<15} {'Capital':<15}")
    print("-" * 60)
    print(f"{'Q-Learning':<20} {results_qlearning['mean_reward']:>6.2f} ± {results_qlearning['std_reward']:<5.2f} "
          f"{results_qlearning['mean_win_rate']*100:>5.1f}% ± {results_qlearning['std_win_rate']*100:<4.1f}% "
          f"${results_qlearning['mean_capital']:>6.2f}")
   
    
    # ===== VISUALIZACIONES =====
    print("\nGenerando gráficas...")
    plot_training_results(metrics_qlearning)
    
    env.close()
    print("\n✓ Entrenamiento completado!")


if __name__ == "__main__":
    main()