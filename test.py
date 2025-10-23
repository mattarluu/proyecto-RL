import AuctionEnv as a

def test_environment():
    """
    Función para testear el entorno con acciones aleatorias.
    """
    print("Testeando el entorno de subastas...\n")
    
    # Crear entorno
    env = a.AuctionEnv(n_auctions=5, render_mode='human')
    
    # Verificar con gymnasium checker
    from gymnasium.utils.env_checker import check_env
    print("Verificando cumplimiento de API Gymnasium...")
    check_env(env, skip_render_check=True)
    print("✓ Entorno cumple con la API de Gymnasium\n")
    
    # Ejecutar un episodio con acciones aleatorias
    print("Ejecutando episodio de prueba con acciones aleatorias...\n")
    
    obs, info = env.reset(seed=42)
    print(f"Observación inicial: {obs}")
    print(f"Info inicial: {info}\n")
    
    total_reward = 0
    done = False
    step_count = 0
    
    while not done:
        action = env.action_space.sample()
        action_names = ['NO_PUJAR', 'BAJA', 'MEDIA', 'ALTA', 'ALL_IN']
        print(f"\n>>> Paso {step_count + 1}: Agente elige {action_names[action]}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        done = terminated or truncated
        step_count += 1
        
        print(f"Recompensa este paso: {reward:.2f}")
        print(f"Recompensa acumulada: {total_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("EPISODIO TERMINADO")
    print("=" * 60)
    print(f"Recompensa total: {total_reward:.2f}")
    print(f"Ítems ganados por agente: {info['agent_items_won']}")
    print(f"Ítems ganados por oponente: {info['opponent_items_won']}")
    print(f"Capital final agente: ${info['agent_capital']:.2f}")
    print(f"Capital final oponente: ${info['opponent_capital']:.2f}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    # Ejecutar test del entorno
    test_environment()