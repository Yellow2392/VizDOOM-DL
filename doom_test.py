import vizdoom as vzd
import numpy as np
import cv2
import time
from collections import deque 
from stable_baselines3 import PPO

MODEL_PATH = "doom_fase1_rocket"
CONFIG_PATH = "VizDOOM-DL/scenarios/rocket_basic.cfg"

# Las 6 acciones idénticas a tu clase VizDoomGym
ACTIONS_LIST = [
    [1, 0, 0, 0, 0, 0], # 0: Fwd (En nuestro gym la acción 0 era IDLE, pero usamos identity matrix)
    [0, 1, 0, 0, 0, 0], # 1: Back
    [0, 0, 1, 0, 0, 0], # 2: Left
    [0, 0, 0, 1, 0, 0], # 3: Right
    [0, 0, 0, 0, 1, 0], # 4: Attack
    [0, 0, 0, 0, 0, 1]  # 5: Use
]
# NOTA: Para no fallar con los índices, armamos la matriz identidad igual que en el entrenamiento
ACTIONS_LIST = np.identity(6, dtype=np.uint8).tolist()

print(f"Cargando {MODEL_PATH}...")
model = PPO.load(MODEL_PATH)

game = vzd.DoomGame()
game.load_config(CONFIG_PATH)
game.set_window_visible(True)  
game.set_mode(vzd.Mode.ASYNC_PLAYER) 
game.init()

# MEMORIA MULTIMODAL (4 frames y 4 estados)
stacked_frames = deque([np.zeros((64, 64, 1), dtype=np.uint8) for _ in range(4)], maxlen=4)
stacked_states = deque([np.zeros(2, dtype=np.float32) for _ in range(4)], maxlen=4)

def get_multimodal_observation(game_instance):
    state = game_instance.get_state()
    if state is None: return None
    
    # 1. Procesar Pantalla
    frame = state.screen_buffer
    if frame.shape[0] == 3: 
        frame = np.transpose(frame, (1, 2, 0))
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img_resized = cv2.resize(img_gray, (64, 64))
    img_final = np.reshape(img_resized, (64, 64, 1))
    
    # 2. Procesar Sensores Internos
    health = game_instance.get_game_variable(vzd.GameVariable.HEALTH)
    ammo = game_instance.get_game_variable(vzd.GameVariable.AMMO2)
    current_state = np.array([health, ammo], dtype=np.float32)
    
    # Añadir a las colas
    stacked_frames.append(img_final)
    stacked_states.append(current_state)
    
    # SB3 concatena VecFrameStack en el último eje
    # Pantallas pasan de (64,64,1) -> (64,64,4)
    # Estados pasan de (2,) -> (8,)
    obs_dict = {
        "pantalla": np.concatenate(stacked_frames, axis=-1),
        "estado": np.concatenate(stacked_states, axis=-1)
    }
    return obs_dict

print("Iniciando Presentación...")
for i in range(5):
    game.new_episode()
    
    # Llenar la memoria inicial con el primer instante del juego
    for _ in range(4): 
        get_multimodal_observation(game)

    while not game.is_episode_finished():
        # Obtener diccionario de observación apilado
        obs = get_multimodal_observation(game)
        if obs is None: break
        
        # Inferencia con el cerebro multimodal
        action_index, _ = model.predict(obs, deterministic=True) 
        
        # Ejecutar acción
        game.make_action(ACTIONS_LIST[int(action_index)])
        
        # Un sleep de 0.028s da unos 35 FPS, se ve muy fluido para el profesor
        time.sleep(0.028)