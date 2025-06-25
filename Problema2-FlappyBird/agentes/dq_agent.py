from agentes.base import Agent
import numpy as np
from collections import defaultdict
import pickle

class QAgent(Agent):
    """
    Agente de Q-Learning mejorado para Flappy Bird.
    Incluye mejoras en discretización del estado y estrategias de exploración.
    """
    def __init__(self, actions, game=None, learning_rate=0.1, discount_factor=0.95,
                 epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01, load_q_table_path="flappy_birds_q_table.pkl"):
        super().__init__(actions, game)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Cargar Q-table existente o crear nueva
        if load_q_table_path:
            try:
                with open(load_q_table_path, 'rb') as f:
                    q_dict = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
                print(f"Q-table cargada desde {load_q_table_path}")
            except FileNotFoundError:
                print(f"Archivo Q-table no encontrado en {load_q_table_path}. Se inicia una nueva Q-table vacía.")
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        else:
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
        
        # Parámetros de discretización optimizados
        self.horizontal_bins = 10  # Reducido para simplificar
        self.vertical_bins = 12   # Optimizado para mejor granularidad
        self.velocity_bins = 6    # Aumentado para mejor control de velocidad
        
        # Contador de visitas para aprendizaje adaptivo
        self.state_visits = defaultdict(int)

    def discretize_state(self, state):
        """
        Discretiza el estado continuo en un estado discreto (tupla).
        """
        # Extraer variables del estado
        player_y = state['player_y']
        player_vel = state['player_vel']
        next_pipe_dist = state['next_pipe_dist_to_player']
        next_pipe_top = state['next_pipe_top_y']
        next_pipe_bottom = state['next_pipe_bottom_y']
        
        # Calcular centro del hueco entre tubos
        gap_center = (next_pipe_top + next_pipe_bottom) / 2.0
        vertical_diff = player_y - gap_center
        
        # Limitar diferencia vertical a un rango manejable
        vertical_diff = max(-300, min(300, vertical_diff))
        
        # Discretizar distancia horizontal con bins más inteligentes
        # Más resolución cuando está cerca del tubo
        if next_pipe_dist < 50:
            bin_x = min(int(next_pipe_dist / 10), 4)  # 5 bins para distancias cortas
        else:
            bin_x = 5 + min(int((next_pipe_dist - 50) / 40), self.horizontal_bins - 6)
        
        # Discretización vertical mejorada con enfoque en zona crítica
        if abs(vertical_diff) <= 40:  # Zona crítica expandida
            # Mayor resolución en zona crítica (6 bins en ±40px)
            bin_y = int((vertical_diff + 40) * 6 / 80)
            bin_y = max(0, min(5, bin_y))
        elif vertical_diff > 40:
            # Zona superior (3 bins)
            bin_y = 6 + min(int((vertical_diff - 40) / 87), 2)
        else:
            # Zona inferior (3 bins)
            bin_y = 9 + min(int((-vertical_diff - 40) / 87), 2)
            
        # Discretizar velocidad vertical con mejor granularidad
        velocity_range = (-12, 12)  # Rango expandido
        vel_clamped = max(velocity_range[0], min(velocity_range[1], player_vel))
        bin_size = (velocity_range[1] - velocity_range[0]) / self.velocity_bins
        vel_bin = min(int((vel_clamped - velocity_range[0]) / bin_size), self.velocity_bins - 1)
            
        return (bin_x, bin_y, vel_bin)

    def act(self, state):
        """
        Elige una acción usando epsilon-greedy mejorado con exploración dirigida.
        """
        # Discretizar el estado
        discrete_state = self.discretize_state(state)
        
        # Incrementar contador de visitas
        self.state_visits[discrete_state] += 1
        
        # Exploración adaptiva: más exploración en estados poco visitados
        visit_bonus = 1.0 / (1.0 + self.state_visits[discrete_state] * 0.1)
        effective_epsilon = self.epsilon + visit_bonus * 0.1
        
        # Exploración: acción aleatoria
        if np.random.random() < effective_epsilon:
            return np.random.choice(self.actions)
        
        # Explotación: mejor acción según Q-table
        q_values = self.q_table[discrete_state]
        
        # Agregar ruido pequeño para romper empates
        noise = np.random.normal(0, 0.001, len(q_values))
        q_values_noisy = q_values + noise
        
        max_action_idx = np.argmax(q_values_noisy)
        return self.actions[max_action_idx]

    def update(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table con aprendizaje adaptivo mejorado.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        
        # Inicializar si el estado no está en la Q-table
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(len(self.actions))
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(len(self.actions))
        
        # Ajuste dinámico de learning rate basado en visitas
        visit_count = self.state_visits[discrete_state]
        adaptive_lr = self.lr / (1 + visit_count * 0.01)
        
        # Modificar recompensa para casos especiales
        modified_reward = reward
        
        # Bonificación por supervivencia en situaciones difíciles
        player_y = state['player_y']
        next_pipe_top = state['next_pipe_top_y']
        next_pipe_bottom = state['next_pipe_bottom_y']
        gap_center = (next_pipe_top + next_pipe_bottom) / 2.0
        
        if abs(player_y - gap_center) < 30 and not done:
            modified_reward += 0.1  # Pequeña bonificación por estar centrado
        
        # Actualización Q-Learning
        current_q = self.q_table[discrete_state][action_idx]
        max_future_q = 0
        if not done:
            max_future_q = np.max(self.q_table[discrete_next_state])
        new_q = current_q + adaptive_lr * (modified_reward + self.gamma * max_future_q - current_q)
        self.q_table[discrete_state][action_idx] = new_q

    def decay_epsilon(self):
        """
        Disminuye epsilon para reducir la exploración con el tiempo.
        """
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_q_table(self, path):
        """
        Guarda la Q-table en un archivo usando pickle.
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table guardada en {path}")

    def load_q_table(self, path):
        """
        Carga la Q-table desde un archivo usando pickle.
        """
        import pickle
        try:
            with open(path, 'rb') as f:
                q_dict = pickle.load(f)
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), q_dict)
            print(f"Q-table cargada desde {path}")
        except FileNotFoundError:
            print(f"Archivo Q-table no encontrado en {path}. Se inicia una nueva Q-table vacía.")
            self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))
