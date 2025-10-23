def watch_trained_agent(agent_path: str, n_episodes: int = 1, delay: float = 1.0):
    """
    Observa al agente entrenado jugando.
    
    Args:
        agent_path: Ruta al agente guardado (.pkl)
        n_episodes: Número de episodios a ver
        delay: Segundos de pausa entre acciones
    """
    import time
    from AuctionEnv import AuctionEnv
    from agent import QLearningAgent as a
    
    # Cargar agente entrenado
    print(f"Cargando agente desde: {agent_path}")
    
    # Detectar tipo de agente por el nombre del archivo
    if 'qlearning' in agent_path.lower():
        agent = a(action_space_size=5)
    
    
    agent.load(agent_path)
    print(f"✓ Agente cargado. Epsilon actual: {agent.epsilon:.3f}")
    
    # Crear entorno con renderizado
    env = AuctionEnv(n_auctions=10, render_mode='human', opponent_strategy='smart')
    
    action_names = ['NO PUJAR', 'BAJA (10%)', 'MEDIA (20%)', 'ALTA (35%)', 'ALL-IN (50%)']
    
    for episode in range(n_episodes):
        print("\n" + "="*70)
        print(f"🎮 EPISODIO {episode + 1}/{n_episodes} - AGENTE ENTRENADO EN ACCIÓN")
        print("="*70)
        
        state, info = env.reset()
        env.render()
        
        total_reward = 0
        done = False
        step = 0
        
        input("\n▶️  Presiona ENTER para comenzar...")
        
        while not done:
            step += 1
            
            # Agente elige acción (sin exploración)
            action = agent.get_action(state, training=False)
            
            print(f"\n{'>'*70}")
            print(f"⚡ SUBASTA #{step}")
            print(f"{'>'*70}")
            print(f"🤖 El agente decide: {action_names[action]}")
            
            # Mostrar Q-values para este estado
            state_key = tuple(state)
            q_values = agent.Q[state_key]
            print(f"\n📊 Q-values del agente para este estado:")
            for i, q_val in enumerate(q_values):
                marker = "👉" if i == action else "  "
                print(f"  {marker} {action_names[i]:<15}: {q_val:>8.2f}")
            
            time.sleep(delay)
            
            # Ejecutar acción
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            env.render()
            
            total_reward += reward
            
            print(f"\n💰 Recompensa obtenida: {reward:>+7.2f}")
            print(f"💰 Recompensa acumulada: {total_reward:>+7.2f}")
            
            state = next_state
            
            if not done:
                time.sleep(delay)
        
        # Resultados finales
        print("\n" + "="*70)
        print("🏁 RESULTADO FINAL DEL EPISODIO")
        print("="*70)
        print(f"📈 Recompensa total: {total_reward:>+8.2f}")
        print(f"🏆 Ítems ganados: {info['agent_items_won']}/{env.n_auctions} ({info['agent_items_won']/env.n_auctions*100:.1f}%)")
        print(f"💵 Capital final: ${info['agent_capital']:.2f}")
        print(f"💸 Total gastado: ${info['total_agent_spent']:.2f}")
        print(f"🤖 Oponente ganó: {info['opponent_items_won']}/{env.n_auctions}")
        print(f"💰 Oponente capital: ${info['opponent_capital']:.2f}")
        
        # Análisis de eficiencia
        if info['agent_items_won'] > 0:
            avg_spent_per_item = info['total_agent_spent'] / info['agent_items_won']
            print(f"📊 Gasto promedio por ítem ganado: ${avg_spent_per_item:.2f}")
        
        print("="*70)
    
    env.close()


# Uso:
if __name__ == "__main__":
    watch_trained_agent('qlearning_agent.pkl', n_episodes=3, delay=1.5)