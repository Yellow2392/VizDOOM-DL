import vizdoom as vzd
import numpy as np
import cv2
import time
from collections import deque 
from stable_baselines3 import PPO

MODEL_PATH = "doom_ppo_final" 
CONFIG_PATH = "scenarios/defend_the_center.cfg"

#! Debe ser IDÉNTICA a la de la clase VizDoomGym
ACTIONS_LIST = [
    [0, 0, 0, 0, 0, 0], # Wait
    [1, 0, 0, 0, 0, 0], # Fwd
    [0, 1, 0, 0, 0, 0], # Back
    [0, 0, 1, 0, 0, 0], # Left
    [0, 0, 0, 1, 0, 0], # Right
    [0, 0, 0, 0, 1, 0], # Attack
    [0, 0, 0, 0, 0, 1]  # Use
]

print(f"Cargando {MODEL_PATH}...")
model = PPO.load(MODEL_PATH)

game = vzd.DoomGame()
game.load_config(CONFIG_PATH)
game.set_window_visible(True)  
game.set_mode(vzd.Mode.ASYNC_PLAYER) 
game.init()

#! memoria para frames
stacked_frames = deque([np.zeros((64, 64), dtype=np.uint8) for _ in range(4)], maxlen=4)

def get_stacked_observation(frame):
    # Preprocesar frame actual
    if frame.shape[0] == 3: frame = np.transpose(frame, (1, 2, 0))
    frame = cv2.resize(frame, (64, 64))
    if len(frame.shape) == 3: frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Añadir a la cola (elimina el más viejo)
    stacked_frames.append(frame)
    
    # Apila (64, 64, 4)
    stack = np.stack(stacked_frames, axis=2)
    return stack

print("Iniciando...")
for i in range(5):
    game.new_episode()
    # Llenar la pila inicial con el primer frame repetido
    state = game.get_state()
    frame0 = state.screen_buffer
    for _ in range(4): get_stacked_observation(frame0) # Reset de memoria

    while not game.is_episode_finished():
        state = game.get_state()
        
        # Obtener observación apilada (4 frames)
        obs = get_stacked_observation(state.screen_buffer)
        
        # Inferencia
        action_index, _ = model.predict(obs, deterministic=True) 
        
        game.make_action(ACTIONS_LIST[int(action_index)])
        time.sleep(0.02)