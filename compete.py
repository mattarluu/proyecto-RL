"""
Script to make the trained agent compete against different opponents
"""

import numpy as np
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

from auctionEnv import AuctionEnv
import matplotlib.pyplot as plt


def compete_vs_opponent(agent, opponent_strategy='adaptive', n_matches=100, verbose=False):
    """
    Competing the trained agent against a specific opponent.
    Args:
        agent
        opponent_strategy:'random', 'fixed', 'adaptive', 'aggressive'
        n_matches: Number of matches to play
        verbose: Whether to print details of each match
    Returns:
        dict: Competition statistics
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
    print(f"COMPETITION: Agent vs Opponent '{opponent_strategy.upper()}'")
    print(f"{'='*70}")
    print(f"Number of games: {n_matches}\n")
    
    for match in range(n_matches):
        state, info = env.reset()
        total_reward = 0
        done = False
        
        if verbose and match < 3:
            print(f"\n{'='*60}")
            print(f"MATCH{match + 1}")
            print(f"{'='*60}")
        
        while not done:
            action = agent.choose_action(state, training=False)
            
            next_state, reward, done, truncated, info = env.step(action)
            
            if verbose and match < 3:
                bid_multipliers = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0, 1.25]
    
                last_round_info = info['history'][-1]
                agent_bid_actual = last_round_info['agent_bid']
                opponent_bid_actual = last_round_info['opponent_bid']
                multiplier_pct = bid_multipliers[action] * 100
                
                print(f"\nRound {int(state[0])}: Item Value ${state[3]:.0f}")
                print(f"  Agent (Action {action}):   {multiplier_pct:.0f}% of the value -> Bid ${agent_bid_actual:.2f}")
                print(f"  Opponent bid:                            ${opponent_bid_actual:.2f}")


            total_reward += reward
            state = next_state
        
        rewards.append(total_reward)
        agent_totals.append(info['agent_total_value'])
        opponent_totals.append(info['opponent_total_value'])
        
        if info['agent_total_value'] > info['opponent_total_value']:
            agent_wins += 1
            result = "VICTORY"
        elif info['agent_total_value'] < info['opponent_total_value']:
            opponent_wins += 1
            result = "DEFEAT"
        else:
            ties += 1
            result = "DRAW"
        
        if verbose and match < 3:
            print(f"\n{result}!")
            print(f"Agente: ${info['agent_total_value']:.0f} | Opponent: ${info['opponent_total_value']:.0f}")
        
        #progress each 10 games
        if (match + 1) % 10 == 0:
            current_win_rate = (agent_wins / (match + 1)) * 100
            print(f"Progress: {match + 1}/{n_matches} matches - Win rate: {current_win_rate:.1f}%")
    
    #statistics
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
    

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Agent victories: {agent_wins} ({stats['agent_win_rate']:.1f}%)")
    print(f"Opponent victories: {opponent_wins} ({(opponent_wins/n_matches)*100:.1f}%)")
    print(f"Draws: {ties} ({(ties/n_matches)*100:.1f}%)")
    print(f"\nAverage reward: {stats['avg_reward']:.2f}")
    print(f"Agent average value: ${stats['avg_agent_total']:.0f}")
    print(f"opponent average value: ${stats['avg_opponent_total']:.0f}")
    print(f"Average margen: ${stats['avg_margin']:.0f}")
    
    #winner
    print(f"\n{'='*70}")
    if agent_wins > opponent_wins:
        print("¡THE AGENT IS THE WINNER!")
        print(f"The agent won {agent_wins - opponent_wins} matches more than the opponent")
    elif opponent_wins > agent_wins:
        print("The opponent won the competition")
        print(f"The opponent won {opponent_wins - agent_wins} matches more than the opponent")
    else:
        print("DRAW - Same number of victories")
    print(f"{'='*70}")
    
    return stats


def tournament(agent, n_matches_per_opponent=100):
    """
    Tournament where the agent competes against all types of opponents.

    Args:
        agent: Trained agent
        n_matches_per_opponent: Matches against each opponent

    Returns:
        dict: Tournament results
    """
    strategies = ['random', 'fixed', 'adaptive', 'aggressive']
    results = {}
    
    print("\n" + "="*70)
    print("TOURNAMENT: Agent vs All opponents")
    print("="*70)
    
    for strategy in strategies:
        stats = compete_vs_opponent(agent, strategy, n_matches_per_opponent, verbose=False)
        results[strategy] = stats
    
    print("\n" + "="*70)
    print("TOURNAMENT SUMMARY")
    print("="*70)
    
    total_wins = sum(r['agent_wins'] for r in results.values())
    total_matches = n_matches_per_opponent * len(strategies)
    overall_win_rate = (total_wins / total_matches) * 100
    
    print(f"\nGlobal result: {total_wins}/{total_matches} victories ({overall_win_rate:.1f}%)\n")
    
    for strategy in strategies:
        stats = results[strategy]
        print(f"{strategy.upper():>12}: {stats['agent_wins']:>3}/{n_matches_per_opponent} victories "
              f"({stats['agent_win_rate']:>5.1f}%) | "
              f"Margen: ${stats['avg_margin']:>6.0f}")
    
    return results



def watch_match(agent, opponent_strategy='adaptive'):
    """
    Watch a complete game between the agent and the opponent step by step.

    Args:
        agent: Trained agent
        opponent_strategy: Opponent's strategy
    """
    env = AuctionEnv(initial_capital=100, render_mode='human')
    env.set_opponent_strategy(opponent_strategy)
    
    print("\n" + "="*70)
    print(f"LIVE MATCH: RL Agent vs Opponent '{opponent_strategy.upper()}'")
    print("="*70)
    
    state, info = env.reset(seed=42)
    env.render()
    
    done = False
    total_reward = 0
    round_num = 1
    
    while not done:
        print(f"\n{'─'*70}")
        print(f"Thinking... (Round {round_num}/10)")
        
        action = agent.choose_action(state, training=False)
        bid_multipliers = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0, 1.25]
        multiplier = bid_multipliers[action]
        item_value = state[3]
        capital = state[1]
        calculated_bid = item_value * multiplier
        actual_bid = min(calculated_bid, capital)

        print(f"Agent decides to bid: {multiplier*100:.0f}% of the value of the item")
        print(f"  (Calculated: ${calculated_bid:.2f}, Real: ${actual_bid:.2f} of ${capital:.2f} available)")
        
        input("\n▶ Press Enter to execute the bid...")
        
        #ejecutar
        next_state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"\nReward received: {reward:.2f}")
        
        if not done:
            env.render()
        
        state = next_state
        round_num += 1
    
    #final result
    print("\n" + "="*70)
    print("Game Over")
    print("="*70)
    
    if info['is_winning']:
        print("\n¡THE AGENT IS THE WINNER!")
    else:
        print("\nThe agent has lost - The opponent wins")
    
    print(f"\nFinal Puntuation:")
    print(f"  Agent:   ${info['agent_total_value']:.2f}")
    print(f"  Opponent: ${info['opponent_total_value']:.2f}")
    
    margin = info['agent_total_value'] - info['opponent_total_value']
    if margin > 0:
        print(f"  Margen:   +${margin:.2f} in favor of the agent")
    else:
        print(f"  Margen:   ${abs(margin):.2f} in favor of the opponent")
    
    print(f"\nTotal reward: {total_reward:.2f}")


def main(training_method='multi', n_episodes=2000):
    """
    Args:
        training_method: 'original', 'multi', 'curriculum', 'adaptive'
        n_episodes
    """
    print("="*70)
    print("🏆 COMPETENCIA DE AGENTE RL EN SUBASTAS")
    print("="*70)

    
    print(f"\nTraining agent with method: '{training_method}'")
    
    if training_method == 'multi':
        print("   Using MULTI-OPPONENT training...")
        
        from trainAuction import train_agent_multi_opponent
        agent, _, _, _ = train_agent_multi_opponent(
            n_episodes=n_episodes,
            rotate_strategy=True,
            render_frequency=200,
            episodes_per_rotation=1000
        )
    
    elif training_method == 'curriculum':
        print("  Using CURRICULUM LEARNING training...")
        from trainAuction import train_agent_curriculum
        agent, _, _ = train_agent_curriculum(
            n_episodes_per_stage=n_episodes // 6
        )
    
    
    #CODE TO PLAY DIFFERENT MODES - UNCOMMENT ON EXECUTE
    
    #OPCIÓN 1: WATCH A LIVE MATCH
    #watch_match(agent, opponent_strategy='fixed')

    
    #OPCION 2: Compete multiple times against a SINGLE opponent
    #compete_vs_opponent(agent, opponent_strategy='adaptive', n_matches=50, verbose=True)
   
    #OPCION 3: TOURNAMENT AGAINST ALL OPPONENTS
    tournament(agent)
    
    


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