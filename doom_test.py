import torch
import torch.nn as nn
import cv2
import numpy as np
import vizdoom as vzd
import time

# Arquitectura (parecida a la de entrenamiento)

class DoomAgentNet(nn.Module):
    def __init__(self, input_h, input_w, num_actions):
        super(DoomAgentNet, self).__init__()
        input_size = input_h * input_w 
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.net(x)

# Configuracion

MODEL_PATH = "doom_model_gpu.pth"  # Modelo entrenado
CONFIG_PATH = "scenarios/D3_battle.cfg" # Escenario entrenado
NUM_ACCIONES = 14  # Debe coincidir con el entrenamiento

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Corriendo inferencia en: {device}")

# Inicializamos la red vacía (64x64 es lo que se usó)
model = DoomAgentNet(64, 64, NUM_ACCIONES).to(device)

# Cargamos los pesos entrenados
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Poner en modo evaluación (congela dropout, etc)
print("¡Cerebro cargado correctamente!")

# Inicia el juegoo
game = vzd.DoomGame()
game.load_config(CONFIG_PATH)
game.set_window_visible(True) # Para visualización
game.set_mode(vzd.Mode.ASYNC_PLAYER) # Modo jugador (no entrenamiento)
game.init()

print("Iniciando partida... Presiona Ctrl+C para parar.")

episodios = 5
for i in range(episodios):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        
        # preprocesar (obtenemos imagen original)
        frame = state.screen_buffer # Viene como (3, H, W) o (H, W, 3) dependiendo config
        
        if frame.shape[0] == 3: 
            frame = np.transpose(frame, (1, 2, 0))
            
        # Resize a 64x64 y Grayscale
        frame = cv2.resize(frame, (64, 64))
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
        # Normalizar y Tensor
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0) # (1 batch, 1 canal, 64, 64)
        frame_tensor = frame_tensor.to(device)

        # INFERENCIA
        with torch.no_grad(): 
            outputs = model(frame_tensor)
            
            # El modelo devuelve valores continuos (0.02, 0.95, -0.1)
            # Necesitamos convertirlos a botones (0 o 1)
            # Usamos un umbral (threshold) de 0.5
            prediction = (outputs > 0.5).float().cpu().numpy()[0]
            
            action = prediction.astype(int).tolist()

        # Action!
        game.make_action(action)
        
        time.sleep(0.02) 

    print(f"Episodio {i+1} terminado.")

game.close()