import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class AuctionEnv(gym.Env):
    """
    El agente compite contra un oponente en una serie de subastas secuenciales.
    Objetivo: Maximizar el beneficio total ganando ítems valiosos y gestionando el capital.
    
    Estados (S):
        - mi_capital_bin: Capital del agente discretizado [0-20, 21-40, 41-60, 61-80, 81-100, 100+]
        - capital_oponente_bin: Capital del oponente discretizado [0-20, 21-40, 41-60, 61-80, 81-100, 100+]
        - valor_item: Valor del ítem en subasta {bajo=15, medio=30, alto=45, muy_alto=60}
        - items_restantes: Número de subastas restantes [0-10]
        - ultimo_ganador: Quién ganó la última subasta {yo=0, oponente=1, ninguno=2}
    
    Acciones (A):
        0: No pujar (puja = 0)
        1: Puja baja (10% del capital)
        2: Puja media (20% del capital)
        3: Puja alta (35% del capital)
        4: All-in (50% del capital)
    
    Recompensas (R):
        - Ganar subasta: valor_item - puja_realizada
        - Perder subasta: 0
        - Penalización por desperdiciar: si (puja - puja_oponente) > 5, penaliza 0.1 * diferencia
        - Bonus final: capital_restante * 0.05
    
    Espacio de estados: 6 × 6 × 4 × 11 × 3 = 4,752 estados
    Espacio de acciones: 5 acciones
    """
    
    metadata = {
        'render_modes': ['human', 'ansi'],
        'render_fps': 4
    }
    
    # Constantes del entorno
    ITEM_VALUES = [15, 30, 45, 60]  # bajo, medio, alto, muy_alto
    ITEM_PROBABILITIES = [0.3, 0.35, 0.25, 0.1]
    CAPITAL_BINS = [20, 40, 60, 80, 100]  # Límites para discretización
    
    def __init__(
        self,
        n_auctions: int = 10,
        initial_capital: float = 100.0,
        render_mode: Optional[str] = None,
        opponent_strategy: str = 'smart'
    ):
        """
        Args:
            n_auctions: Número de subastas por episodio
            initial_capital: Capital inicial para ambos jugadores
            render_mode: Modo de renderizado ('human', 'ansi', None)
            opponent_strategy: Estrategia del oponente ('random', 'smart', 'greedy')
        """
        super().__init__()
        
        self.n_auctions = n_auctions
        self.initial_capital = initial_capital
        self.render_mode = render_mode
        self.opponent_strategy = opponent_strategy
        self.observation_space = spaces.MultiDiscrete([
            6,   # mi_capital_bin: 0-20, 21-40, 41-60, 61-80, 81-100, 100+
            6,   # capital_oponente_bin: igual discretización
            4,   # valor_item: bajo(15), medio(30), alto(45), muy_alto(60)
            11,  # items_restantes: 0, 1, 2, ..., 10
            3    # ultimo_ganador: yo(0), oponente(1), ninguno(2)
        ])
        
        # Definición del espacio de acciones (discreto)
        self.action_space = spaces.Discrete(5)
        
        # Variables de estado (se inicializan en reset)
        self.agent_capital = 0.0
        self.opponent_capital = 0.0
        self.current_auction = 0
        self.current_item_value = 0
        self.last_winner = 2  # 0=agente, 1=oponente, 2=ninguno
        
        # Variables para tracking (info)
        self.last_agent_bid = 0.0
        self.last_opponent_bid = 0.0
        self.agent_items_won = 0
        self.opponent_items_won = 0
        self.total_agent_spent = 0.0
        self.total_opponent_spent = 0.0
    
    def _discretize_capital(self, capital: float) -> int:
        """
        Convierte capital continuo a bin discreto.
        
        Args:
            capital: Cantidad de capital continua
            
        Returns:
            Índice del bin (0-5)
        """
        for i, threshold in enumerate(self.CAPITAL_BINS):
            if capital <= threshold:
                return i
        return 5  # 100+
    
    def _get_obs(self) -> np.ndarray:
        """
        Obtiene la observación actual del entorno.
        
        Returns:
            Array numpy con el estado discretizado
        """
        return np.array([
            self._discretize_capital(self.agent_capital),
            self._discretize_capital(self.opponent_capital),
            self.ITEM_VALUES.index(self.current_item_value),
            self.n_auctions - self.current_auction,  # items restantes
            self.last_winner
        ], dtype=np.int32)
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Obtiene información adicional del entorno (para debugging y análisis).
        
        Returns:
            Diccionario con información del estado actual
        """
        return {
            'agent_capital': float(self.agent_capital),
            'opponent_capital': float(self.opponent_capital),
            'current_item_value': int(self.current_item_value),
            'auctions_remaining': int(self.n_auctions - self.current_auction),
            'agent_bid': float(self.last_agent_bid),
            'opponent_bid': float(self.last_opponent_bid),
            'last_winner': int(self.last_winner),
            'agent_items_won': int(self.agent_items_won),
            'opponent_items_won': int(self.opponent_items_won),
            'total_agent_spent': float(self.total_agent_spent),
            'total_opponent_spent': float(self.total_opponent_spent)
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reinicia el entorno al estado inicial.
        
        Args:
            seed: Semilla para reproducibilidad
            options: Opciones adicionales (no usado)
            
        Returns:
            Tupla (observación_inicial, info)
        """
        super().reset(seed=seed)
        
        # Resetear capitales
        self.agent_capital = self.initial_capital
        self.opponent_capital = self.initial_capital
        
        # Resetear contador de subastas
        self.current_auction = 0
        
        # Resetear ganadores
        self.last_winner = 2  # Ninguno al inicio
        
        # Resetear estadísticas
        self.agent_items_won = 0
        self.opponent_items_won = 0
        self.total_agent_spent = 0.0
        self.total_opponent_spent = 0.0
        self.last_agent_bid = 0.0
        self.last_opponent_bid = 0.0
        
        # Generar primer ítem
        self.current_item_value = self.np_random.choice(
            self.ITEM_VALUES,
            p=self.ITEM_PROBABILITIES
        )
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _action_to_bid(self, action: int, capital: float) -> float:
        """
        Convierte una acción a cantidad de puja.
        
        Args:
            action: Índice de acción (0-4)
            capital: Capital disponible
            
        Returns:
            Cantidad a pujar
        """
        if capital <= 0:
            return 0.0
        
        if action == 0:  # No pujar
            return 0.0
        elif action == 1:  # Puja baja (10%)
            return min(capital * 0.1, capital)
        elif action == 2:  # Puja media (20%)
            return min(capital * 0.2, capital)
        elif action == 3:  # Puja alta (35%)
            return min(capital * 0.35, capital)
        else:  # All-in (50%)
            return min(capital * 0.5, capital)
    
    def _opponent_policy(self) -> int:
        """
        Política del oponente (estocástica).
        
        Returns:
            Acción del oponente (0-4)
        """
        if self.opponent_strategy == 'random':
            return self.np_random.integers(0, 5)
        
        elif self.opponent_strategy == 'greedy':
            # Siempre puja proporcional al valor del ítem
            if self.current_item_value >= 60:
                return 4  # all-in
            elif self.current_item_value >= 45:
                return 3  # alta
            elif self.current_item_value >= 30:
                return 2  # media
            else:
                return 1  # baja
        
        else:  # 'smart' (default)
            # Estrategia inteligente considerando capital
            if self.opponent_capital <= 10:
                # Con poco capital, muy conservador
                return self.np_random.choice([0, 1], p=[0.7, 0.3])
            
            # Decisión basada en valor del ítem
            if self.current_item_value >= 60:  # muy alto
                return self.np_random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            elif self.current_item_value >= 45:  # alto
                return self.np_random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            elif self.current_item_value >= 30:  # medio
                return self.np_random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
            else:  # bajo
                return self.np_random.choice([0, 1], p=[0.6, 0.4])
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Acción a ejecutar (0-4)
            
        Returns:
            Tupla (observación, recompensa, terminado, truncado, info)
        """
        # Obtener pujas
        agent_bid = self._action_to_bid(action, self.agent_capital)
        opponent_action = self._opponent_policy()
        opponent_bid = self._action_to_bid(opponent_action, self.opponent_capital)
        
        # Guardar para info y análisis
        self.last_agent_bid = agent_bid
        self.last_opponent_bid = opponent_bid
        
        # Resolver subasta
        reward = 0.0
        
        if agent_bid > opponent_bid:
            # Agente gana la subasta
            self.agent_capital -= agent_bid
            self.total_agent_spent += agent_bid
            self.agent_items_won += 1
            reward = self.current_item_value - agent_bid
            self.last_winner = 0
            
            # Penalización por desperdiciar dinero
            waste = agent_bid - opponent_bid
            if waste > 5:
                reward -= waste * 0.1
        
        elif opponent_bid > agent_bid:
            # Oponente gana la subasta
            self.opponent_capital -= opponent_bid
            self.total_opponent_spent += opponent_bid
            self.opponent_items_won += 1
            reward = 0.0
            self.last_winner = 1
        
        else:  # Empate (ambas pujas iguales)
            # Desempate aleatorio 50-50
            if self.np_random.random() < 0.5:
                # Agente gana
                self.agent_capital -= agent_bid
                self.total_agent_spent += agent_bid
                self.agent_items_won += 1
                reward = self.current_item_value - agent_bid
                self.last_winner = 0
            else:
                # Oponente gana
                self.opponent_capital -= opponent_bid
                self.total_opponent_spent += opponent_bid
                self.opponent_items_won += 1
                reward = 0.0
                self.last_winner = 1
        
        # Avanzar a siguiente subasta
        self.current_auction += 1
        
        # Verificar si terminó el episodio
        terminated = self.current_auction >= self.n_auctions
        
        if terminated:
            # Bonus por capital restante al final del episodio
            reward += self.agent_capital * 0.05
        else:
            # Generar siguiente ítem
            self.current_item_value = self.np_random.choice(
                self.ITEM_VALUES,
                p=self.ITEM_PROBABILITIES
            )
        
        # Obtener nueva observación
        observation = self._get_obs()
        info = self._get_info()
        
        # Renderizar si está configurado
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info
    
    def render(self):
        """
        Renderiza el estado actual del entorno.
        """
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "ansi":
            return self._render_ansi()
    
    def _render_human(self):
        """Renderizado en consola (human mode)."""
        print("\n" + "=" * 60)
        print(f"{'SUBASTA':<20} {self.current_auction}/{self.n_auctions}")
        print("=" * 60)
        
        if self.current_auction <= self.n_auctions:
            print(f"{'Ítem en Subasta:':<25} Valor = ${self.current_item_value}")
            print("-" * 60)
        
        print(f"{'AGENTE':<25} {'OPONENTE':<25}")
        print(f"{'Capital:':<15} ${self.agent_capital:>8.2f}   {'Capital:':<15} ${self.opponent_capital:>8.2f}")
        print(f"{'Ítems ganados:':<15} {self.agent_items_won:>8}   {'Ítems ganados:':<15} {self.opponent_items_won:>8}")
        
        if hasattr(self, 'last_agent_bid') and self.current_auction > 0:
            print("-" * 60)
            print(f"{'Última Puja:':<15} ${self.last_agent_bid:>8.2f}   {'Última Puja:':<15} ${self.last_opponent_bid:>8.2f}")
            
            winner_names = ["AGENTE ✓", "OPONENTE ✓", "NINGUNO"]
            winner_str = winner_names[self.last_winner]
            print(f"\n{'Ganador de la última subasta:':<30} {winner_str}")
        
        print("=" * 60)
    
    def _render_ansi(self) -> str:
        """
        Renderizado en formato string (ansi mode).
        
        Returns:
            String con el estado del entorno
        """
        output = []
        output.append("=" * 60)
        output.append(f"SUBASTA {self.current_auction}/{self.n_auctions}")
        output.append("=" * 60)
        
        if self.current_auction <= self.n_auctions:
            output.append(f"Ítem: Valor ${self.current_item_value}")
        
        output.append(f"Capital Agente: ${self.agent_capital:.2f}")
        output.append(f"Capital Oponente: ${self.opponent_capital:.2f}")
        
        if hasattr(self, 'last_agent_bid') and self.current_auction > 0:
            output.append(f"Última puja agente: ${self.last_agent_bid:.2f}")
            output.append(f"Última puja oponente: ${self.last_opponent_bid:.2f}")
            winner_names = ["AGENTE", "OPONENTE", "NINGUNO"]
            output.append(f"Ganador: {winner_names[self.last_winner]}")
        
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def close(self):
        """Limpia recursos (no necesario en este entorno)."""
        pass

    

