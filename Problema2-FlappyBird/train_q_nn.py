import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_data
import os
import matplotlib.pyplot as plt


# --- Constantes y Configuración ---
QTABLE_PATH = 'flappy_birds_q_table.pkl'  # Cambia el path si es necesario
MODEL_SAVE_PATH = 'flappy_q_nn_model/flappy_birds_dq_nn_model.keras'
NUM_ACTIONS = 2 # Nada (0), Saltar (1)

# Hiperparámetros de entrenamiento
EPOCHS = 80 
BATCH_SIZE = 64 # Se podría probar bajar a 32 o 16 
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42 


# --- Cargar Q-table entrenada ---

print("\nCargando la Q-table...")
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)["q_table"]
    

if len(q_table) == 0:
    print("Error: La Q-table está vacía. No se puede entrenar.")
    exit()

# --- PREPARACION DE LOS DATOS PARA ENTRENAMIENTO ---


# --- 1. Conversión de la Q-table en X (estados) e y (valores Q para cada acción)

X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(q_values)
    
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("Error: El dataset cargado no contiene muestras.")
    exit()


# --- 2. Shuffle de datos
print("\nBarajando los datos...")
X, y = shuffle_data(X, y, random_state=RANDOM_STATE)
print("Datos barajados.")

# Calculamos el numero de features
num_features = X.shape[1]

# Mostramos algunos valores
print(f"Total de muestras cargadas inicialmente: {len(X)}")
print(f"Número de características de entrada detectadas: {num_features}")
print(f"Forma de los estados (X) después de procesar: {X.shape}")
print(f"Forma de las acciones (y) después de procesar: {y.shape}")


# --- 3. Dibisión de los datos en conjuntos de entrenamiento y validación
print(f"\nDividiendo datos en entrenamiento y validación (split: {VALIDATION_SPLIT})...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_STATE, # Usar el mismo random_state para consistencia
)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del conjunto de validación: {X_val.shape[0]}")

# --- 4. Definicion de la red neuronal 
print("\nDefiniendo el modelo de red neuronal...")

model = keras.Sequential([
    layers.Input(shape=(num_features,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    # layers.Dense(8, activation='relu'),
    layers.Dense(NUM_ACTIONS, activation='linear')
])
model.summary()


# --- 5. Compilación del modelo
model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mse'])

# --- 6. Entrenamiento del modelo

# --- Callbacks ---
PACIENCIA = 10
UMBRAL_ACC = 0.7

#---
# EarlyStopping
#---
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PACIENCIA,
    verbose=1,
    mode="min"
)

#---
# ModelCheckpoint
#---
CHECKPOINT_PATH = 'checkpoints/flappy_birds_dq_model_best.keras'
checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
if not os.path.exists(checkpoint_dir) and checkpoint_dir: # Asegurarse que checkpoint_dir no sea vacío
    os.makedirs(checkpoint_dir)

model_checkpoint_callback = ModelCheckpoint(
    filepath=CHECKPOINT_PATH, # Guardará el modelo completo (no solo pesos)
    save_weights_only=False,  # Guardar el modelo completo
    monitor='val_loss',   # Métrica a monitorear
    mode='min',               # Guardar cuando la métrica monitoreada sea máxima
    save_best_only=True,      # Solo guardar el mejor modelo
    verbose=1
)

#---
# ReduceLROnPlateau
#---
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss', # Monitorear la pérdida de validación
    factor=0.2,         # Factor por el cual reducir LR: new_lr = lr * factor
    patience=5,         # Número de épocas sin mejora después de las cuales se reduce LR
    min_lr=0.0001,      # Límite inferior para la tasa de aprendizaje
    verbose=1
)

callbacks_list = [model_checkpoint_callback, reduce_lr_callback]

print(f"\nEntrenando el modelo durante {EPOCHS} épocas con batch_size {BATCH_SIZE}...")
try:
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list,
                        verbose=1)
except KeyboardInterrupt:
    print("Error: falla en el entrenamiento del modelo")
    exit()

# --- 7. Métricas de entrenamiento del modelo
losses = history.history['loss']
val_losses = history.history['val_loss']
print(f"Mejor época: {np.argmin(losses)} con loss = {min(losses):.4f}")
print(f"Mejor época: {np.argmin(val_losses)} con val_loss = {min(val_losses):.4f}")


mse_train = history.history['mse']
mse_val = history.history['val_mse']
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs_range = range(len(mse_train))

plt.figure(figsize=(10, 5))

# MSE
plt.plot(epochs_range, mse_train, label='MSE Entrenamiento')
plt.plot(epochs_range, mse_val, label='MSE Validación')
plt.legend(loc='upper right')
plt.title('Error Cuadrático Medio (MSE) - Entrenamiento y Validación')
plt.xlabel('Época')
plt.ylabel('MSE')

# Guardar el gráfico
plt.savefig("dqlearning_flappybird_entrenamiento.png")

plt.tight_layout()
plt.show()

# --- Mostrar resultados del entrenamiento ---
# Completar: Imprimir métricas de entrenamiento PREGUNTAR SI METRICAS O GRAFICO O AMBOS ???

# --- Guardar el modelo entrenado ---
# COMPLETAR: Cambia el nombre si lo deseas

# Verificacion carpeta existente si la carpeta no esta creada
folder = os.path.dirname(MODEL_SAVE_PATH)
if not os.path.exists(folder) and folder != '':
    os.makedirs(folder)

model.save(MODEL_SAVE_PATH)
print('Modelo guardado como TensorFlow SavedModel en flappy_q_nn_model/')

# --- Notas para los alumnos ---
# - Puedes modificar la arquitectura de la red y los hiperparámetros.
# - Puedes usar la red entrenada para aproximar la Q-table y luego usarla en un agente tipo DQN.
# - Si tu estado es una tupla de enteros, no hace falta normalizar, pero puedes probarlo.
# - Si tienes dudas sobre cómo usar el modelo para predecir acciones, consulta la documentación de Keras.
