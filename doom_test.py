import vizdoom as vzd
import numpy as np
import cv2
import time
from stable_baselines3 import PPO

MODEL_PATH = "doom_ppo_final" 

CONFIG_PATH = "scenarios/defend_the_center.cfg" 

#! Debe ser IDÉNTICA a la que usaste en la clase VizDoomGym
ACTIONS_LIST = [
    [0, 0, 0, 0, 0, 0], # 0. Quieto
    [1, 0, 0, 0, 0, 0], # 1. Avanzar
    [0, 1, 0, 0, 0, 0], # 2. Girar Izq
    [0, 0, 1, 0, 0, 0], # 3. Girar Der
    [0, 0, 0, 1, 0, 0], # 4. Disparar
    [0, 0, 0, 0, 1, 0]  # 5. ABRIR PUERTA (USE)
]

print(f"Cargando modelo desde {MODEL_PATH}...")

model = PPO.load(MODEL_PATH)
print("¡Cerebro cargado correctamente!")

game = vzd.DoomGame()
game.load_config(CONFIG_PATH)
game.set_window_visible(True)  
game.set_mode(vzd.Mode.ASYNC_PLAYER) 
game.init()

print("Iniciando partida... Presiona Ctrl+C para parar.")

episodios = 5

for i in range(episodios):
    game.new_episode()
    
    while not game.is_episode_finished():
        state = game.get_state()
        frame = state.screen_buffer # Viene (3, H, W) o (H, W, 3)

        #! PREPROCESAMIENTO: Debe ser IGUAL al de VizDoomGym)
        # formato para opencv (H, W, Canales)
        if frame.shape[0] == 3:
            frame = np.transpose(frame, (1, 2, 0))
        
        frame = cv2.resize(frame, (64, 64))
        
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
        frame = frame[:, :, np.newaxis]

        #! Inferenciaa
        action_index, _ = model.predict(frame, deterministic=True)

        print(f"Predicción: {action_index}")
        
        action_buttons = ACTIONS_LIST[int(action_index)]

        #! Actuaar
        game.make_action(action_buttons)
        
        time.sleep(0.03) 

    print(f"Episodio {i+1} terminado. Score: {game.get_total_reward()}")

game.close()