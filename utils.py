import numpy as np
def plot_training_results(metrics_q):
    """
    Genera gráficas comparativas del entrenamiento.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Función helper para suavizar curvas
    def moving_average(data, window=100):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 1. Recompensas
    ax = axes[0, 0]
    ax.plot(moving_average(metrics_q['episode_rewards']), label='Q-Learning', alpha=0.8)
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Recompensa (promedio móvil 100)')
    ax.set_title('Curvas de Aprendizaje')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Win Rate
    ax = axes[0, 1]
    ax.plot(moving_average(metrics_q['win_rates']), label='Q-Learning', alpha=0.8)
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Win Rate (promedio móvil 100)')
    ax.set_title('Tasa de Victoria')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Epsilon decay
    ax = axes[1, 0]
    ax.plot(metrics_q['epsilon_history'], label='Q-Learning', alpha=0.8)
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Epsilon')
    ax.set_title('Decay de Epsilon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Capital final
    ax = axes[1, 1]
    ax.plot(moving_average(metrics_q['capital_finals']), label='Q-Learning', alpha=0.8)
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Capital Final (promedio móvil 100)')
    ax.set_title('Gestión de Capital')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica guardada: training_comparison.png")
    plt.show()