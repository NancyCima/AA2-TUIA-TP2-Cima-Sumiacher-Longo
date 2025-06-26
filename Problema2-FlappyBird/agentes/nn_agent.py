from agentes.base import Agent
import numpy as np
import tensorflow as tf
from agentes.dq_agent import QAgent  # Para usar la función discretize_state
import time

class NNAgent(Agent):
    """
    Agente que utiliza una red neuronal entrenada para aproximar la Q-table.
    La red debe estar guardada como TensorFlow SavedModel.
    """
    def __init__(self, actions, game=None, model_path='flappy_q_nn_model/flappy_birds_dq_nn_model.keras'):
        super().__init__(actions, game)
        # Cargar el modelo entrenado
        self.model = tf.keras.models.load_model(model_path)

        # Creamos un agente QAgent para reutilizar discretización
        self.q_agent = QAgent(actions)


    def act(self, state):
        """
        Debe transformar el estado al formato de entrada de la red y devolver la acción con mayor Q.
        """
        # t0 = time.time()
        # 1. Discretizar el estado (usar la función de QAgent)
        discrete_state = self.q_agent.discretize_state(state)
        # t1 = time.time()
        # 2. Convertir la tupla a array para la red
        state_input = np.array(discrete_state).reshape(1, -1)
        # t2 = time.time()
        # print(f"Discretización: {t1 - t0:.6f}s, Predicción: {t2 - t1:.6f}s")
        # 3. Predecir Q-values con la red neuronal
        q_values = self.model.predict(state_input, verbose=0)[0]
        
        # 4. Elegir la acción con mayor Q-value
        action_index = np.argmax(q_values)
        
        # 5. Retornar la acción correspondiente
        return self.actions[action_index]