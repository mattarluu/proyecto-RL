"""
Script para hacer competir al agente entrenado contra diferentes oponentes
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from auctionEnv import AuctionEnv
import matplotlib.pyplot as plt


def compete_vs_opponent(agent, opponent_strategy='adaptive', n_matches=100, verbose=False):
    """
    Hace competir al agente entrenado contra un oponente específico.
    
    Args:
        agent: Agente entrenado con Q-Learning
        opponent_strategy: Estrategia del oponente ('random', 'fixed', 'adaptive', 'aggressive')
        n_matches: Número de partidas a jugar
        verbose: Si imprimir detalles de cada partida
    
    Returns:
        dict: Estadísticas de la competencia
    """
    env = AuctionEnv(initial_capital=100)
    env.set_opponent_strategy(opponent_strategy)
    
    agent_wins = 0
    opponent_wins = 0
    ties = 0
    rewards = []
    agent_totals = []
    opponent_totals = []
    
    print(f"\n{'='*70}")
    print(f"COMPETENCIA: Agente RL vs Oponente '{opponent_strategy.upper()}'")
    print(f"{'='*70}")
    print(f"Número de partidas: {n_matches}\n")
    
    for match in range(n_matches):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        if verbose and match < 3:
            print(f"\n{'='*60}")
            print(f"PARTIDA {match + 1}")
            print(f"{'='*60}")
        
        while not done:
            # Agente elige acción (sin exploración, solo explotación)
            action = agent.choose_action(state, training=False)
            
            # Ejecutar acción
            next_state, reward, done, truncated, info = env.step(action)
            
            if verbose and match < 3:
                # Replicamos la lista de multiplicadores que definimos en auctionEnv.py
                bid_multipliers = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0, 1.25]
                
                # Obtenemos las pujas reales del historial (que está en 'info')
                # Hacemos esto DESPUÉS de env.step()
                last_round_info = info['history'][-1]
                agent_bid_actual = last_round_info['agent_bid']
                opponent_bid_actual = last_round_info['opponent_bid']
                
                # Imprimimos la acción (intención) y la puja real (resultado)
                multiplier_pct = bid_multipliers[action] * 100
                
                print(f"\nRonda {int(state[0])}: Item valor ${state[3]:.0f}")
                print(f"  Agente (Acción {action}):   {multiplier_pct:.0f}% del valor -> Puja ${agent_bid_actual:.2f}")
                print(f"  Oponente puja:                            ${opponent_bid_actual:.2f}")
                # -------------------------

            total_reward += reward
            state = next_state
        
        # Registrar resultados
        rewards.append(total_reward)
        agent_totals.append(info['agent_total_value'])
        opponent_totals.append(info['opponent_total_value'])
        
        if info['agent_total_value'] > info['opponent_total_value']:
            agent_wins += 1
            result = "VICTORIA"
        elif info['agent_total_value'] < info['opponent_total_value']:
            opponent_wins += 1
            result = "DERROTA"
        else:
            ties += 1
            result = "EMPATE"
        
        if verbose and match < 3:
            print(f"\n{result}!")
            print(f"Agente: ${info['agent_total_value']:.0f} | Oponente: ${info['opponent_total_value']:.0f}")
        
        # Progreso cada 10 partidas
        if (match + 1) % 10 == 0:
            current_win_rate = (agent_wins / (match + 1)) * 100
            print(f"Progreso: {match + 1}/{n_matches} partidas - Win rate: {current_win_rate:.1f}%")
    
    # Calcular estadísticas
    stats = {
        'agent_wins': agent_wins,
        'opponent_wins': opponent_wins,
        'ties': ties,
        'agent_win_rate': (agent_wins / n_matches) * 100,
        'avg_reward': np.mean(rewards),
        'avg_agent_total': np.mean(agent_totals),
        'avg_opponent_total': np.mean(opponent_totals),
        'avg_margin': np.mean(np.array(agent_totals) - np.array(opponent_totals))
    }
    
    # Imprimir resultados
    print(f"\n{'='*70}")
    print("RESULTADOS FINALES")
    print(f"{'='*70}")
    print(f"Victorias del Agente: {agent_wins} ({stats['agent_win_rate']:.1f}%)")
    print(f"Victorias del Oponente: {opponent_wins} ({(opponent_wins/n_matches)*100:.1f}%)")
    print(f"Empates: {ties} ({(ties/n_matches)*100:.1f}%)")
    print(f"\nRecompensa promedio: {stats['avg_reward']:.2f}")
    print(f"Valor promedio Agente: ${stats['avg_agent_total']:.0f}")
    print(f"Valor promedio Oponente: ${stats['avg_opponent_total']:.0f}")
    print(f"Margen promedio: ${stats['avg_margin']:.0f}")
    
    # MOSTRAR GANADOR CLARAMENTE
    print(f"\n{'='*70}")
    if agent_wins > opponent_wins:
        print("🏆🎉 ¡EL AGENTE ES EL GANADOR! 🎉🏆")
        print(f"El agente ganó {agent_wins - opponent_wins} partidas más que el oponente")
    elif opponent_wins > agent_wins:
        print("😞 El oponente ganó la competencia")
        print(f"El oponente ganó {opponent_wins - agent_wins} partidas más que el agente")
    else:
        print("🤝 EMPATE - Mismo número de victorias")
    print(f"{'='*70}")
    
    return stats


def tournament(agent, n_matches_per_opponent=100):
    """
    Torneo donde el agente compite contra todos los tipos de oponentes.
    
    Args:
        agent: Agente entrenado
        n_matches_per_opponent: Partidas contra cada oponente
    
    Returns:
        dict: Resultados del torneo
    """
    strategies = ['random', 'fixed', 'adaptive', 'aggressive']
    results = {}
    
    print("\n" + "="*70)
    print("🏆 TORNEO: Agente vs Todos los Oponentes")
    print("="*70)
    
    for strategy in strategies:
        stats = compete_vs_opponent(agent, strategy, n_matches_per_opponent, verbose=False)
        results[strategy] = stats
    
    # Resumen del torneo
    print("\n" + "="*70)
    print("📊 RESUMEN DEL TORNEO")
    print("="*70)
    
    total_wins = sum(r['agent_wins'] for r in results.values())
    total_matches = n_matches_per_opponent * len(strategies)
    overall_win_rate = (total_wins / total_matches) * 100
    
    print(f"\nResultado Global: {total_wins}/{total_matches} victorias ({overall_win_rate:.1f}%)\n")
    
    for strategy in strategies:
        stats = results[strategy]
        print(f"{strategy.upper():>12}: {stats['agent_wins']:>3}/{n_matches_per_opponent} victorias "
              f"({stats['agent_win_rate']:>5.1f}%) | "
              f"Margen: ${stats['avg_margin']:>6.0f}")
    
    return results


def plot_tournament_results(results):
    """
    Grafica los resultados del torneo.
    
    Args:
        results: Diccionario con resultados del torneo
    """
    strategies = list(results.keys())
    win_rates = [results[s]['agent_win_rate'] for s in strategies]
    margins = [results[s]['avg_margin'] for s in strategies]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de win rates
    colors = ['#2ecc71' if wr >= 50 else '#e74c3c' for wr in win_rates]
    bars1 = ax1.bar(strategies, win_rates, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='50% (equilibrio)')
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('Tasa de Victoria del Agente vs Cada Oponente', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, wr in zip(bars1, win_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{wr:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Gráfica de márgenes
    colors2 = ['#2ecc71' if m >= 0 else '#e74c3c' for m in margins]
    bars2 = ax2.bar(strategies, margins, color=colors2, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Margen Promedio ($)', fontsize=12)
    ax2.set_title('Margen de Victoria Promedio (Agente - Oponente)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for bar, m in zip(bars2, margins):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${m:.0f}',
                ha='center', va=va, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/tournament_results.png', dpi=150, bbox_inches='tight')
    print("\n📈 Gráfica guardada en: tournament_results.png")
    plt.close()


def watch_match(agent, opponent_strategy='adaptive'):
    """
    Observa una partida completa entre el agente y el oponente paso a paso.
    
    Args:
        agent: Agente entrenado
        opponent_strategy: Estrategia del oponente
    """
    env = AuctionEnv(initial_capital=100, render_mode='human')
    env.set_opponent_strategy(opponent_strategy)
    
    print("\n" + "="*70)
    print(f"🎮 PARTIDA EN VIVO: Agente RL vs Oponente '{opponent_strategy.upper()}'")
    print("="*70)
    
    state, info = env.reset(seed=42)
    env.render()
    
    done = False
    total_reward = 0
    round_num = 1
    
    while not done:
        print(f"\n{'─'*70}")
        print(f"Pensando... (Ronda {round_num}/10)")
        
        # Agente decide
        action = agent.choose_action(state, training=False)
        # Replicar la lista de multiplicadores para imprimir
        bid_multipliers = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0, 1.25]
        multiplier = bid_multipliers[action]

        # Obtener info del estado
        item_value = state[3]
        capital = state[1]
        calculated_bid = item_value * multiplier
        actual_bid = min(calculated_bid, capital)

        print(f"✓ Agente decide pujar: {multiplier*100:.0f}% del VALOR DEL ITEM")
        print(f"  (Calculado: ${calculated_bid:.2f}, Real: ${actual_bid:.2f} de ${capital:.2f} disponibles)")
        
        input("\n▶ Presiona Enter para ejecutar la puja...")
        
        # Ejecutar
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\n💰 Recompensa recibida: {reward:.2f}")
        
        if not done:
            env.render()
        
        state = next_state
        round_num += 1
    
    # Resultado final
    print("\n" + "="*70)
    print("🏁 PARTIDA FINALIZADA")
    print("="*70)
    
    if info['is_winning']:
        print("\n🎉🏆 ¡EL AGENTE HA GANADO! 🏆🎉")
    else:
        print("\n😞💔 El agente ha perdido - El oponente gana")
    
    print(f"\nPuntuación Final:")
    print(f"  🤖 Agente:   ${info['agent_total_value']:.2f}")
    print(f"  👤 Oponente: ${info['opponent_total_value']:.2f}")
    
    margin = info['agent_total_value'] - info['opponent_total_value']
    if margin > 0:
        print(f"  📊 Margen:   +${margin:.2f} a favor del agente")
    else:
        print(f"  📊 Margen:   ${abs(margin):.2f} a favor del oponente")
    
    print(f"\n💎 Recompensa total: {total_reward:.2f}")


def main(training_method='multi', n_episodes=2000):
    """
    Función principal.
    
    Args:
        training_method: 'original', 'multi', 'curriculum', 'adaptive'
        n_episodes: Número de episodios de entrenamiento
    """
    print("="*70)
    print("🏆 COMPETENCIA DE AGENTE RL EN SUBASTAS")
    print("="*70)
    
    # Opción 1: Cargar un agente ya entrenado (si lo tienes guardado)
    # agent = load_agent('agent.pkl')
    
    # Opción 2: Entrenar un nuevo agente con el método elegido
    print(f"\n🔄 Entrenando agente con método: '{training_method}'")
    
    if training_method == 'multi':
        print("   Usando entrenamiento MULTI-OPONENTE...")
        
        from trainBetter import train_agent_multi_opponent
        agent, _, _, _ = train_agent_multi_opponent(
            n_episodes=n_episodes,
            rotate_strategy=True,
            render_frequency=200
        )
    
    elif training_method == 'curriculum':
        print("   Usando entrenamiento CURRICULUM LEARNING...")
        from trainBetter import train_agent_curriculum
        agent, _, _ = train_agent_curriculum(
            n_episodes_per_stage=n_episodes // 4
        )

    
    
    # Opción 3: Ver una partida en vivo
    #print("\n" + "="*70)
    #print("OPCIÓN 1: VER UNA PARTIDA EN VIVO")
    #print("="*70)
    #watch_match(agent, opponent_strategy='adaptive')
    
    #Opción 4: Competir múltiples veces contra un oponente
    #print("\n" + "="*70)
    #print("OPCIÓN 2: COMPETENCIA MÚLTIPLE")
    #print("="*70)
    compete_vs_opponent(agent, opponent_strategy='adaptive', n_matches=50, verbose=True)
    print("\n" + "="*70)
    #print("OPCIÓN 2: COMPETENCIA MÚLTIPLE")
    #print("="*70)
    compete_vs_opponent(agent, opponent_strategy='adaptive', n_matches=50, verbose=True)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Competencia de Agente RL')
    parser.add_argument('--metodo', type=str, default='multi',
                       choices=['original', 'multi', 'curriculum', 'adaptive'],
                       help='Método de entrenamiento a usar')
    parser.add_argument('--episodios', type=int, default=2000,
                       help='Número de episodios de entrenamiento')
    args = parser.parse_args()
    
    main(training_method=args.metodo, n_episodes=args.episodios)