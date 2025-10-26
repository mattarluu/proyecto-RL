#Auction Environment
"""
The game consists of 10 auctions where the agent and an opponent bid on items.
The player with the highest total value (items + remaining capital) at the end wins.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AuctionEnv(gym.Env):
    """
    - State space: 6 dimensions (round, agent_capital, opponent_capital, item_value, agent_items, opponent_items)
    - Action space: Discrete(7), 0%, 25%...
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, initial_capital=100, render_mode=None):
        super().__init__()
        
        self.initial_capital = initial_capital
        self.render_mode = render_mode
        
        self.bid_multipliers = [
            0.0,    
            0.25,  
            0.5,    
            0.6,    
            0.75,   
            1.0,    
            1.25    
        ]

        self.action_space = spaces.Discrete(len(self.bid_multipliers))
        
        #observation space
        #[round, agent_capital, opponent_capital, item_value, agent_items, opponent_items]
        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([10, initial_capital*2, initial_capital*2, 100, 1000, 1000], dtype=np.float32),
            dtype=np.float32
        )
        
        #it can be modified
        self.opponent_strategy = 'adaptive'  #'random', 'fixed', 'adaptive', 'aggressive'
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        #initial state
        self.round = 1
        self.agent_capital = self.initial_capital
        self.opponent_capital = self.initial_capital
        self.agent_items_value = 0
        self.opponent_items_value = 0
        
        #first item (between 5 - 50)
        self.current_item_value = self._generate_item_value()
        
        #history, only for information
        self.history = []
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Returns:
            observation: new state
            reward
            terminated
            truncated
            info: aditional information
        """
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action}")
        
        #based on the action, calculate de bid
        bid_multiplier = self.bid_multipliers[action]
        calculated_bid = self.current_item_value * bid_multiplier
        agent_bid = min(calculated_bid, self.agent_capital)
        
        #opponnent bid
        opponent_bid = self._opponent_bid()
        
        reward = 0  
        winner = 'tie' 
        
        if agent_bid > opponent_bid:
            self.agent_items_value += self.current_item_value
            self.agent_capital -= agent_bid
    
            reward = self.current_item_value - agent_bid  # Base
            
            #BONUS  action 3 (60% - better with opponent 'fixed')
            #if action == 3:
               #reward += 25
            
            #BONUS for eficienci 
            if agent_bid < self.current_item_value * 0.65:
                reward += 10
            
            #PENALTYfor waste 
            minimum_winning_bid = opponent_bid + 0.01
            waste = agent_bid - minimum_winning_bid
            rounds_left = 10 - self.round + 1
            if rounds_left <= 3 and agent_bid < self.current_item_value * 0.5:
                reward += 15
            if waste > 0:
                # Si quedan muchas rondas, el desperdicio es más grave
                if rounds_left > 5:
                    waste_factor = 1.0  # Penaliza más (70% del waste)
                elif rounds_left > 2:
                    waste_factor = 0.7  # Penaliza moderado
                else:
                    waste_factor = 0.3  # Al final, puedes gastar más
                
                waste_penalty = -(waste * waste_factor)
                reward += waste_penalty
            
            winner = 'agent'

        elif opponent_bid > agent_bid:
            self.opponent_items_value += self.current_item_value
            self.opponent_capital -= opponent_bid
            
            #PENALTY adjusted by value
            if self.current_item_value > 30:
                reward = -20  # Penaliza MÁS perder items valiosos
            elif self.current_item_value > 20:
                reward = -10  #medium item
            else:
                reward = 0    #cheap item
            
            #EXTRA PENALTY if we could have won but didn't try
            if action == 0 and self.agent_capital >= self.current_item_value * 0.6:
                reward -= 15
            
            winner = 'opponent'
        
        else:
            #draw
            reward = 0
            winner = 'tie'
        
        #save in history -- informative
        self.history.append({
            'round': self.round,
            'item_value': self.current_item_value,
            'agent_bid': agent_bid,
            'opponent_bid': opponent_bid,
            'winner': winner,
            'reward': reward
        })
        
        self.round += 1
        terminated = False
        
        if self.round > 10:
            terminated = True
        
            #final reward: WIN or LOSE
            agent_total = self.agent_items_value + self.agent_capital
            opponent_total = self.opponent_items_value + self.opponent_capital
            
            if agent_total > opponent_total:
                margin = agent_total - opponent_total
                reward += 200 + margin  
            elif agent_total < opponent_total:
                margin = opponent_total - agent_total
                reward -= 200 + margin  
            
        else:
            #new item
            self.current_item_value = self._generate_item_value()
        
        observation = self._get_observation()
        info = self._get_info()
        
        
        return observation, reward, terminated, False, info
    
    def _opponent_bid(self):
        """
        Opponent bid based on strategy
        Returns:
            float: Amount bid by the opponent
        """
        if self.opponent_strategy == 'random':
            bid_percentage = np.random.uniform(0, 1)
            return self.opponent_capital * bid_percentage
        
        elif self.opponent_strategy == 'fixed':
            #always bids the 50% of the item, if has capital
            bid = min(self.current_item_value * 0.5, self.opponent_capital)
            return bid
        
        elif self.opponent_strategy == 'adaptive':
            rounds_left = 11 - self.round
            if rounds_left <= 0:
                return 0
            
            # Análisis de situación
            item_ratio = self.current_item_value / 50.0
            urgency = 1.0 - (rounds_left / 10.0)
            
            # ¿Estoy ganando o perdiendo?
            opponent_total = self.opponent_items_value + self.opponent_capital
            agent_total = self.agent_items_value + self.agent_capital
            score_diff = opponent_total - agent_total
            
            if score_diff < -50:
                desperation = 0.3
            elif score_diff < 0:
                desperation = 0.15
            else:
                desperation = 0.0
            
            # ¿El agente tiene poco capital?
            if self.agent_capital < self.current_item_value * 0.6:
                opportunism = 0.2
            else:
                opportunism = 0.0
            
            # ¿Tengo suficiente capital?
            avg_capital_per_round = self.opponent_capital / max(rounds_left, 1)
            if avg_capital_per_round < 20:
                conservation = -0.2
            else:
                conservation = 0.0
            
            # Cálculo final
            bid_percentage = (
                0.25 +
                (item_ratio * 0.3) +
                (urgency * 0.15) +
                desperation +
                opportunism +
                conservation
            )
            
            bid_percentage = max(0.1, min(bid_percentage, 1.0))
            
            # IMPORTANTE: Pujar sobre el VALOR del item, no sobre el capital
            calculated_bid = self.current_item_value * bid_percentage
            return min(calculated_bid, self.opponent_capital)
                
        elif self.opponent_strategy == 'aggressive':
            # Estrategia agresiva - puja alto por items buenos
            if self.current_item_value > 30:
                return self.opponent_capital * 0.7
            elif self.current_item_value > 20:
                return self.opponent_capital * 0.5
            else:
                return self.opponent_capital * 0.2
        
        return 0
    
    def _generate_item_value(self):
        """
        Genera el valor de un item de forma aleatoria.
        
        Returns:
            float: Valor del item (entre 5 y 50)
        """
        return np.random.uniform(5, 50)
    
    def _get_observation(self):
        """
        Obtiene el estado actual como observación.
        
        Returns:
            np.array: Vector de estado [ronda, capital_agente, capital_oponente, 
                                       valor_item, items_agente, items_oponente]
        """
        return np.array([
            self.round,
            self.agent_capital,
            self.opponent_capital,
            self.current_item_value,
            self.agent_items_value,
            self.opponent_items_value
        ], dtype=np.float32)
    
    def _get_info(self):
        """
        Obtiene información adicional del estado actual.
        
        Returns:
            dict: Información adicional
        """
        agent_total = self.agent_items_value + self.agent_capital
        opponent_total = self.opponent_items_value + self.opponent_capital
        
        return {
            'round': self.round,
            'agent_total_value': agent_total,
            'opponent_total_value': opponent_total,
            'is_winning': agent_total > opponent_total,
            'history': self.history
        }
    
    def render(self):
        """
        Renderiza el estado actual del environment.
        """
        if self.render_mode == 'human':
            # Si el juego terminó, mostrar resultado final
            if self.round > 10:
                print(f"\n{'='*60}")
                print("🏁 JUEGO TERMINADO")
                print(f"{'='*60}")
                
                agent_total = self.agent_capital + self.agent_items_value
                opponent_total = self.opponent_capital + self.opponent_items_value
                
                print(f"\n📊 PUNTUACIÓN FINAL:")
                print(f"\n🤖 AGENTE:")
                print(f"  Capital restante: ${self.agent_capital:.2f}")
                print(f"  Valor items: ${self.agent_items_value:.2f}")
                print(f"  ➤ TOTAL: ${agent_total:.2f}")
                
                print(f"\n👤 OPONENTE:")
                print(f"  Capital restante: ${self.opponent_capital:.2f}")
                print(f"  Valor items: ${self.opponent_items_value:.2f}")
                print(f"  ➤ TOTAL: ${opponent_total:.2f}")
                
                print(f"\n{'='*60}")
                if agent_total > opponent_total:
                    print("🏆🎉 ¡EL AGENTE GANA! 🎉🏆")
                    margin = agent_total - opponent_total
                    print(f"Victoria por margen de: ${margin:.2f}")
                elif opponent_total > agent_total:
                    print("😞💔 El oponente gana")
                    margin = opponent_total - agent_total
                    print(f"Derrota por margen de: ${margin:.2f}")
                else:
                    print("🤝 EMPATE PERFECTO")
                print(f"{'='*60}\n")
                
            else:
                # Estado normal durante el juego
                print(f"\n{'='*60}")
                print(f"RONDA {self.round}/10")
                print(f"{'='*60}")
                print(f"Item actual: Valor = ${self.current_item_value:.2f}")
                print(f"\nAGENTE:")
                print(f"  Capital: ${self.agent_capital:.2f}")
                print(f"  Valor items: ${self.agent_items_value:.2f}")
                print(f"  TOTAL: ${self.agent_capital + self.agent_items_value:.2f}")
                print(f"\nOPONENTE:")
                print(f"  Capital: ${self.opponent_capital:.2f}")
                print(f"  Valor items: ${self.opponent_items_value:.2f}")
                print(f"  TOTAL: ${self.opponent_capital + self.opponent_items_value:.2f}")
                
                if self.history:
                    last = self.history[-1]
                    print(f"\nÚltima subasta:")
                    print(f"  Agente pujó: ${last['agent_bid']:.2f}")
                    print(f"  Oponente pujó: ${last['opponent_bid']:.2f}")
                    print(f"  Ganador: {last['winner']}")
    
    def close(self):
        """Cierra el environment."""
        pass
    
    def set_opponent_strategy(self, strategy):
        """
        Cambia la estrategia del oponente.
        
        Args:
            strategy: 'random', 'fixed', 'adaptive', 'aggressive'
        """
        valid_strategies = ['random', 'fixed', 'adaptive', 'aggressive']
        if strategy not in valid_strategies:
            raise ValueError(f"Estrategia debe ser una de: {valid_strategies}")
        self.opponent_strategy = strategy


# Función helper para registrar el environment
def register_auction_env():
    """
    Registra el environment en Gymnasium para poder usarlo con gym.make()
    """
    from gymnasium.envs.registration import register
    
    register(
        id='Auction-v0',
        entry_point='auction_env:AuctionEnv',
        max_episode_steps=10,
    )


if __name__ == "__main__":
    # Ejemplo de uso
    print("Ejemplo de uso del Environment de Subastas\n")
    
    env = AuctionEnv(initial_capital=100, render_mode='human')
    
    # Episodio de ejemplo con acciones aleatorias
    observation, info = env.reset()
    env.render()
    
    terminated = False
    total_reward = 0
    
    while not terminated:
        # Acción aleatoria para demostración
        action = env.action_space.sample()
        
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env.render()
        
        if not terminated:
            input("\nPresiona Enter para continuar a la siguiente ronda...")
    
    print(f"\n{'='*60}")
    print(f"JUEGO TERMINADO")
    print(f"{'='*60}")
    print(f"Recompensa total: {total_reward:.2f}")
    print(f"¿Ganó el agente? {info['is_winning']}")
    print(f"Valor final agente: ${info['agent_total_value']:.2f}")
    print(f"Valor final oponente: ${info['opponent_total_value']:.2f}")
    
    env.close()