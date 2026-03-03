from __future__ import annotations

import os
import sys
import time
import json
import curses
import argparse
import threading
import multiprocessing as mp
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Set, Optional, Any

import numpy as np
import pandas as pd

import yaml
from imageio.v2 import imwrite
from pynput import keyboard
import cv2

from doom_controller import DoomController


# Defaults de grabación -----------------------------------------------------------------------------
VIDEO_FPS_DEFAULT        = 35.0
VIDEO_BACKEND_DEFAULT    = "ffmpeg"      # "ffmpeg" | "opencv" | "npz"
VIDEO_CONTAINER_DEFAULT  = "mkv"         # "mkv" | "mp4"
VIDEO_CODEC_DEFAULT      = "libx264"     # ffmpeg: "libx264" | "libx265" | "ffv1" | "prores_ks" | "libvpx-vp9"
VIDEO_CRF_DEFAULT        = 18            # 0 = lossless, 18 se ve bien
VIDEO_PRESET_DEFAULT     = "veryfast"    # x264: ultrafast/veryfast/...
CHUNK_SIZE_DEFAULT       = 350
QUEUE_MAXSIZE_DEFAULT    = 256

# Armas conocidas de Doom (slot -> nombre corto)
WEAPON_NAMES = {
    1: "Fist/Pistol",
    2: "Shotgun    ",
    3: "Chaingun   ",
    4: "Rocket Lncr",
    5: "Plasma Gun ",
    6: "BFG 9000   ",
    7: "Chainsaw   ",
}

SKILL_NAMES = {
    1: "I'm Too Young To Die",
    2: "Hey, Not Too Rough  ",
    3: "Hurt Me Plenty      ",
    4: "Ultra-Violence      ",
    5: "Nightmare!          ",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Estado compartido para el HUD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HUDState:
    """
    Actualizado por el loop del juego (hilo principal).
    Leído por CursesHUD (hilo secundario).
    Protegido por un Lock para evitar torn reads.
    """
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Escenario
    doom_wad:   str = "unknown"
    doom_map:   str = "map01"
    doom_skill: int = 3

    # Variables de juego
    health:               float = 100.0
    armor:                float = 0.0
    killcount:            int   = 0
    selected_weapon:      int   = 1
    selected_weapon_ammo: float = 0.0
    ammo: Dict[int, float] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    weapons_owned: List[bool] = field(default_factory=lambda: [False] * 8)  # índice 1-7

    # Rendimiento
    tics:              int   = 0
    fps:               float = 0.0
    cumulative_reward: float = 0.0

    # Controles activos
    pressed_buttons: List[str] = field(default_factory=list)

    # Grabación
    recording:    bool = False
    session_dir:  str  = ""

    # Estado de la sesión
    running:    bool = True   # False cuando el loop termina → HUD cierra
    end_reason: str  = ""


# ═══════════════════════════════════════════════════════════════════════════════
#  HUD con curses
# ═══════════════════════════════════════════════════════════════════════════════

# Paleta de colores (pares definidos en _init_colors)
C_TITLE    = 1   # rojo sobre negro   — cabecera
C_LABEL    = 2   # cian sobre negro   — etiquetas
C_VALUE    = 3   # blanco sobre negro — valores
C_OK       = 4   # verde sobre negro  — valores positivos (hp alto, kills)
C_WARN     = 5   # amarillo sobre negro — hp medio
C_DANGER   = 6   # rojo sobre negro   — hp bajo
C_REC      = 7   # rojo sobre negro bold — indicador REC
C_DIM      = 8   # negro brillante    — elementos inactivos
C_BORDER   = 9   # azul sobre negro   — bordes
C_KEYON    = 10  # negro sobre verde  — tecla activa
C_KEYOFF   = 11  # blanco sobre negro — tecla inactiva


def _init_colors() -> None:
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_TITLE,  curses.COLOR_RED,     -1)
    curses.init_pair(C_LABEL,  curses.COLOR_CYAN,    -1)
    curses.init_pair(C_VALUE,  curses.COLOR_WHITE,   -1)
    curses.init_pair(C_OK,     curses.COLOR_GREEN,   -1)
    curses.init_pair(C_WARN,   curses.COLOR_YELLOW,  -1)
    curses.init_pair(C_DANGER, curses.COLOR_RED,     -1)
    curses.init_pair(C_REC,    curses.COLOR_RED,     -1)
    curses.init_pair(C_DIM,    curses.COLOR_BLACK,   -1)
    curses.init_pair(C_BORDER, curses.COLOR_BLUE,    -1)
    curses.init_pair(C_KEYON,  curses.COLOR_BLACK,   curses.COLOR_GREEN)
    curses.init_pair(C_KEYOFF, curses.COLOR_WHITE,   -1)


def _health_color(hp: float) -> int:
    if hp > 60:
        return C_OK
    if hp > 30:
        return C_WARN
    return C_DANGER


def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0) -> None:
    """addstr que ignora errores de borde de pantalla."""
    try:
        max_y, max_x = win.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        available = max_x - x - 1
        if available <= 0:
            return
        win.addstr(y, x, text[:available], attr)
    except curses.error:
        pass


def _draw_hline(win, y: int, x: int, width: int, attr: int = 0) -> None:
    _safe_addstr(win, y, x, "─" * width, attr)


def _draw_box_row(win, y: int, left: str, mid: str, right: str, widths: List[int], attr: int = 0) -> None:
    """Dibuja una fila de bordes con separadores en las columnas dadas."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y:
        return
    x = 0
    _safe_addstr(win, y, x, left, attr)
    x += 1
    for i, w in enumerate(widths):
        _safe_addstr(win, y, x, "─" * w, attr)
        x += w
        if i < len(widths) - 1:
            _safe_addstr(win, y, x, mid, attr)
            x += 1
    _safe_addstr(win, y, x, right, attr)


def _render_hud(win: "curses.window", state: HUDState) -> None:
    """Renderiza el panel completo a partir del estado actual."""
    win.erase()
    max_y, max_x = win.getmaxyx()

    # Dimensiones mínimas para no crashear
    if max_y < 20 or max_x < 50:
        _safe_addstr(win, 0, 0, "Terminal demasiado pequeña (mín 50×20)", curses.color_pair(C_DANGER))
        win.refresh()
        return

    ba = curses.color_pair(C_BORDER)           # atributo de borde
    la = curses.color_pair(C_LABEL)            # etiqueta
    va = curses.color_pair(C_VALUE)            # valor
    da = curses.color_pair(C_DIM) | curses.A_DIM

    # Ancho total disponible (dejamos 1 col de margen derecho)
    W = min(max_x - 1, 72)
    col1 = 22   # ancho columna izquierda (sin bordes)
    col2 = W - col1 - 3  # ancho columna derecha

    row = 0

    # ── Cabecera ──────────────────────────────────────────────────────────────
    title = " DOOM  ─  HUD EN VIVO "
    title_x = max(0, (W - len(title)) // 2)
    _safe_addstr(win, row, 0, "╔" + "═" * (W - 2) + "╗", ba)
    row += 1
    _safe_addstr(win, row, 0, "║", ba)
    _safe_addstr(win, row, title_x, title, curses.color_pair(C_TITLE) | curses.A_BOLD)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1
    _draw_box_row(win, row, "╠", "╦", "╣", [col1, col2], ba)
    row += 1

    # ── Fila: Escenario | Estado vital ────────────────────────────────────────
    wad_short = os.path.basename(state.doom_wad).replace(".wad", "")
    skill_name = SKILL_NAMES.get(state.doom_skill, str(state.doom_skill))

    lines_left = [
        ("WAD   ", wad_short[:col1 - 8]),
        ("Mapa  ", state.doom_map),
        ("Skill ", f"{state.doom_skill} · {skill_name[:col1 - 10]}"),
    ]

    hp_color = curses.color_pair(_health_color(state.health))
    lines_right = [
        ("♥  Vida  ", f"{int(state.health):>3}", hp_color),
        ("✦  Armadura ", f"{int(state.armor):>3}", curses.color_pair(C_LABEL)),
        ("☠  Kills ", f"{int(state.killcount):>3}", curses.color_pair(C_OK) | curses.A_BOLD),
    ]

    n_rows_block1 = max(len(lines_left), len(lines_right))
    for i in range(n_rows_block1):
        _safe_addstr(win, row, 0, "║", ba)
        if i < len(lines_left):
            lbl, val = lines_left[i]
            _safe_addstr(win, row, 2, lbl, la)
            _safe_addstr(win, row, 2 + len(lbl), val, va)
        _safe_addstr(win, row, col1 + 1, "║", ba)
        if i < len(lines_right):
            lbl, val, col = lines_right[i]
            _safe_addstr(win, row, col1 + 3, lbl, la)
            _safe_addstr(win, row, col1 + 3 + len(lbl), val, col | curses.A_BOLD)
        _safe_addstr(win, row, W - 1, "║", ba)
        row += 1

    # ── Separador Armas | Munición ─────────────────────────────────────────────
    _draw_box_row(win, row, "╠", "╬", "╣", [col1, col2], ba)
    row += 1

    # Armas disponibles (slots 1-7)
    weapon_line_parts = []
    for slot in range(1, 8):
        owned = state.weapons_owned[slot] if slot < len(state.weapons_owned) else False
        is_selected = (slot == int(state.selected_weapon))
        weapon_line_parts.append((str(slot), owned, is_selected))

    sel_name = WEAPON_NAMES.get(int(state.selected_weapon), f"Arma {int(state.selected_weapon)}")

    ammo_lines = [
        (f"Slot {s} · {WEAPON_NAMES.get(s, '?')}", state.ammo.get(s, 0.0))
        for s in range(1, 5)
    ]

    n_rows_block2 = max(3 + len(ammo_lines), 4)

    # Fila 0: cabeceras
    _safe_addstr(win, row, 0, "║", ba)
    _safe_addstr(win, row, 2, "ARMAS", la | curses.A_BOLD)
    _safe_addstr(win, row, col1 + 1, "║", ba)
    _safe_addstr(win, row, col1 + 3, "MUNICIÓN", la | curses.A_BOLD)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    # Fila 1: slots con colores
    _safe_addstr(win, row, 0, "║", ba)
    cx = 2
    for slot_str, owned, selected in weapon_line_parts:
        if selected:
            attr = curses.color_pair(C_KEYON) | curses.A_BOLD
        elif owned:
            attr = curses.color_pair(C_OK)
        else:
            attr = da
        _safe_addstr(win, row, cx, f"[{slot_str}]", attr)
        cx += 4
    _safe_addstr(win, row, col1 + 1, "║", ba)
    # Primera línea de munición
    lbl, val = ammo_lines[0]
    ammo_val = f"{int(val):>4}" if val > 0 else "  --"
    _safe_addstr(win, row, col1 + 3, lbl, la)
    _safe_addstr(win, row, col1 + 3 + len(lbl) + 1, ammo_val, va)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    # Fila 2: arma seleccionada + ammo actual
    _safe_addstr(win, row, 0, "║", ba)
    _safe_addstr(win, row, 2, f"► {sel_name}", curses.color_pair(C_WARN) | curses.A_BOLD)
    _safe_addstr(win, row, col1 + 1, "║", ba)
    lbl, val = ammo_lines[1]
    ammo_val = f"{int(val):>4}" if val > 0 else "  --"
    _safe_addstr(win, row, col1 + 3, lbl, la)
    _safe_addstr(win, row, col1 + 3 + len(lbl) + 1, ammo_val, va)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    # Fila 3: ammo del arma equipada
    _safe_addstr(win, row, 0, "║", ba)
    _safe_addstr(win, row, 2, f"  Ammo actual: ", la)
    _safe_addstr(win, row, 17, f"{int(state.selected_weapon_ammo):>4}", curses.color_pair(C_WARN) | curses.A_BOLD)
    _safe_addstr(win, row, col1 + 1, "║", ba)
    lbl, val = ammo_lines[2]
    ammo_val = f"{int(val):>4}" if val > 0 else "  --"
    _safe_addstr(win, row, col1 + 3, lbl, la)
    _safe_addstr(win, row, col1 + 3 + len(lbl) + 1, ammo_val, va)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    # Fila 4: ammo[4]
    _safe_addstr(win, row, 0, "║", ba)
    _safe_addstr(win, row, col1 + 1, "║", ba)
    lbl, val = ammo_lines[3]
    ammo_val = f"{int(val):>4}" if val > 0 else "  --"
    _safe_addstr(win, row, col1 + 3, lbl, la)
    _safe_addstr(win, row, col1 + 3 + len(lbl) + 1, ammo_val, va)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    # ── Separador rendimiento ─────────────────────────────────────────────────
    _draw_box_row(win, row, "╠", "╩", "╣", [col1, col2], ba)
    row += 1

    # ── Rendimiento ──────────────────────────────────────────────────────────
    _safe_addstr(win, row, 0, "║", ba)
    _safe_addstr(win, row, 2, "RENDIMIENTO", la | curses.A_BOLD)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    _safe_addstr(win, row, 0, "║", ba)
    perf_str = (
        f"  Tics: {state.tics:>6}   "
        f"FPS: {state.fps:>5.1f}   "
        f"Recompensa: {state.cumulative_reward:>+8.2f}"
    )
    _safe_addstr(win, row, 1, perf_str, va)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    # ── Teclas activas ────────────────────────────────────────────────────────
    _safe_addstr(win, row, 0, "╠" + "═" * (W - 2) + "╣", ba)
    row += 1

    _safe_addstr(win, row, 0, "║", ba)
    _safe_addstr(win, row, 2, "TECLAS: ", la)
    kx = 10
    all_keys = ["W", "A", "S", "D", "UP", "DOWN", "LEFT", "RIGHT",
                "CTRL", "Q", "SPACE", "SHIFT", "1", "2", "3", "4", "5", "6", "7"]
    for key in all_keys:
        active = key in state.pressed_buttons
        attr = curses.color_pair(C_KEYON) | curses.A_BOLD if active else da
        label = f" {key} "
        if kx + len(label) + 1 >= W - 1:
            break
        _safe_addstr(win, row, kx, label, attr)
        kx += len(label) + 1
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    # ── Grabación / Estado final ──────────────────────────────────────────────
    _safe_addstr(win, row, 0, "╠" + "═" * (W - 2) + "╣", ba)
    row += 1

    _safe_addstr(win, row, 0, "║", ba)
    if state.recording:
        rec_label = "● REC"
        _safe_addstr(win, row, 2, rec_label, curses.color_pair(C_REC) | curses.A_BOLD | curses.A_BLINK)
        sess_short = os.path.basename(state.session_dir)
        _safe_addstr(win, row, 9, f" {sess_short}", va)
    else:
        _safe_addstr(win, row, 2, "○ Modo libre  (sin grabación)", da)
    _safe_addstr(win, row, W - 17, "[ESC para salir]", curses.color_pair(C_DIM) | curses.A_DIM)
    _safe_addstr(win, row, W - 1, "║", ba)
    row += 1

    _safe_addstr(win, row, 0, "╚" + "═" * (W - 2) + "╝", ba)

    win.refresh()


class CursesHUD(threading.Thread):
    """
    Hilo daemon que renderiza el HUD con curses a ~10 Hz.
    El hilo principal actualiza `state` y pone `state.running = False` para detenerlo.
    """

    def __init__(self, state: HUDState, refresh_hz: float = 10.0) -> None:
        super().__init__(daemon=True, name="CursesHUD")
        self.state = state
        self._period = 1.0 / max(1.0, refresh_hz)
        self._stdscr: Optional["curses.window"] = None

    def run(self) -> None:
        try:
            curses.wrapper(self._curses_main)
        except Exception:
            pass  # Nunca crashear el hilo del HUD

    def _curses_main(self, stdscr: "curses.window") -> None:
        self._stdscr = stdscr
        curses.curs_set(0)
        stdscr.nodelay(True)
        _init_colors()

        next_t = time.perf_counter()
        while True:
            with self.state.lock:
                running = self.state.running

            _render_hud(stdscr, self.state)

            if not running:
                # Mostrar pantalla de fin durante 3 s antes de cerrar curses
                self._render_end_screen(stdscr)
                time.sleep(3.0)
                break

            next_t += self._period
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()

    def _render_end_screen(self, win: "curses.window") -> None:
        with self.state.lock:
            reason   = self.state.end_reason
            kills    = self.state.killcount
            tics     = self.state.tics
            reward   = self.state.cumulative_reward

        max_y, max_x = win.getmaxyx()
        W = min(max_x - 1, 72)

        messages = {
            "death":   ("  GAME OVER — MORISTE  ",  curses.color_pair(C_DANGER) | curses.A_BOLD),
            "success": ("  VICTORIA — MISION CUMPLIDA  ", curses.color_pair(C_OK) | curses.A_BOLD),
            "timeout": ("  TIEMPO AGOTADO  ",          curses.color_pair(C_WARN)  | curses.A_BOLD),
            "user":    ("  SESION TERMINADA POR EL USUARIO  ", curses.color_pair(C_LABEL) | curses.A_BOLD),
        }
        msg_text, msg_attr = messages.get(reason, ("  SESION FINALIZADA  ", curses.color_pair(C_VALUE)))

        win.erase()
        ba = curses.color_pair(C_BORDER)
        row = max(0, max_y // 2 - 4)

        _safe_addstr(win, row, 0, "╔" + "═" * (W - 2) + "╗", ba)
        row += 1
        cx = max(0, (W - len(msg_text)) // 2)
        _safe_addstr(win, row, 0, "║" + " " * (W - 2) + "║", ba)
        _safe_addstr(win, row, cx, msg_text, msg_attr)
        row += 1
        _safe_addstr(win, row, 0, "╠" + "═" * (W - 2) + "╣", ba)
        row += 1
        stats = [
            f"  Kills    : {kills}",
            f"  Tics     : {tics}",
            f"  Recompensa: {reward:+.2f}",
        ]
        for s in stats:
            _safe_addstr(win, row, 0, "║" + " " * (W - 2) + "║", ba)
            _safe_addstr(win, row, 2, s, curses.color_pair(C_VALUE))
            row += 1
        _safe_addstr(win, row, 0, "╚" + "═" * (W - 2) + "╝", ba)
        win.refresh()


# ═══════════════════════════════════════════════════════════════════════════════
#  Utilidades
# ═══════════════════════════════════════════════════════════════════════════════

def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No se encuentra el archivo YAML: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"El YAML debe ser un mapeo (dict): {path}")
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Teclado
# ═══════════════════════════════════════════════════════════════════════════════

KEY_ALIASES = {
    "ctrl_l": "CTRL", "ctrl_r": "CTRL", "ctrl": "CTRL",
    "shift_l": "SHIFT", "shift_r": "SHIFT", "shift": "SHIFT",
    "space": "SPACE", "esc": "ESCAPE", "enter": "ENTER",
    "return": "ENTER", "up": "UP", "down": "DOWN",
    "left": "LEFT", "right": "RIGHT",
}


def normalize_key(key: keyboard.Key | keyboard.KeyCode) -> Optional[str]:
    try:
        if isinstance(key, keyboard.KeyCode):
            ch = key.char
            if ch is None:
                return None
            return ch.upper() if len(ch) == 1 else ch.upper()
        else:
            name = str(key).split('.')[-1].lower()
            return KEY_ALIASES.get(name, name.upper())
    except Exception:
        return None


class GlobalKeyState:
    def __init__(self) -> None:
        self._pressed: Set[str] = set()
        self._lock = threading.Lock()
        self._exit_requested = False
        self._listener: Optional[keyboard.Listener] = None

    def on_press(self, key) -> None:
        k = normalize_key(key)
        if k is None:
            return
        with self._lock:
            self._pressed.add(k)
            if k == "ESCAPE":
                self._exit_requested = True

    def on_release(self, key) -> None:
        k = normalize_key(key)
        if k is None:
            return
        with self._lock:
            self._pressed.discard(k)

    def start(self) -> None:
        self._listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def snapshot(self) -> Set[str]:
        with self._lock:
            return set(self._pressed)

    def exit_requested(self) -> bool:
        with self._lock:
            return bool(self._exit_requested)

    def clear_exit(self) -> None:
        with self._lock:
            self._exit_requested = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Mapeo de teclas a botones
# ═══════════════════════════════════════════════════════════════════════════════

def build_action_vector(
    button_names: List[str],
    keymap: Dict[str, Any],
    pressed: Set[str],
) -> np.ndarray:
    K = len(button_names)
    vec = np.zeros((K,), dtype=np.int32)
    keyboard_map: Dict[str, str] = {k.upper(): v for k, v in keymap.get("keyboard", {}).items()}
    for key in pressed:
        btn = keyboard_map.get(key)
        if btn is None or btn == "QUIT":
            continue
        try:
            j = button_names.index(btn)
            vec[j] = 1
        except ValueError:
            pass
    return vec


def _pressed_to_button_names(
    button_names: List[str],
    keymap: Dict[str, Any],
    pressed: Set[str],
) -> List[str]:
    """Devuelve los nombres de botones activos (para el HUD de teclas)."""
    keyboard_map: Dict[str, str] = {k.upper(): v for k, v in keymap.get("keyboard", {}).items()}
    active = []
    for key in pressed:
        btn = keyboard_map.get(key)
        if btn and btn != "QUIT":
            active.append(key)
    return list(active)


# ═══════════════════════════════════════════════════════════════════════════════
#  Grabación asíncrona
# ═══════════════════════════════════════════════════════════════════════════════

def _open_ffmpeg_stdin(w: int, h: int, fps: float, codec: str, preset: str, crf: int, out_path: str) -> Any:
    import subprocess
    args = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(float(fps)),
        "-i", "-", "-an", "-threads", "0",
    ]
    if codec in ("libx264", "libx265"):
        args += ["-c:v", codec, "-preset", preset, "-crf", str(int(crf))]
        if codec == "libx265" and int(crf) == 0:
            args += ["-x265-params", "lossless=1"]
    elif codec == "ffv1":
        args += ["-c:v", "ffv1", "-level", "3", "-g", "1"]
    elif codec == "prores_ks":
        args += ["-c:v", "prores_ks", "-profile:v", "3"]
    elif codec == "libvpx-vp9":
        args += ["-c:v", "libvpx-vp9", "-lossless", "1", "-row-mt", "1"]
    else:
        args += ["-c:v", codec]
    args += [out_path]

    try:
        proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    except FileNotFoundError as e:
        raise RuntimeError("No se encontró 'ffmpeg' en el PATH.") from e
    if proc.stdin is None:
        raise RuntimeError("ffmpeg no expuso stdin.")
    return proc


def _writer_process(
    queue: mp.Queue,
    session_dir: str,
    button_names: List[str],
    gamevar_names: List[str],
    doom_wad: str,
    doom_map: str,
    doom_skill: int,
    video_backend: str,
    video_container: str,
    video_codec: str,
    video_crf: int,
    video_preset: str,
    video_fps: float,
    chunk_size: int,
) -> None:
    import traceback

    frames_dir = os.path.join(session_dir, "frames")
    ensure_dir(frames_dir)

    vw = None
    ff_proc = None
    video_path = os.path.join(
        session_dir,
        f"screen.{video_container if video_backend == 'ffmpeg' else 'avi'}"
    )

    if video_backend == "opencv":
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if video_codec.upper() == "XVID" else "MJPG"))

    depth_buf: List[np.ndarray] = []
    labels_buf: List[np.ndarray] = []
    automap_buf: List[np.ndarray] = []
    screen_buf: List[np.ndarray] = []
    chunk_idx = 0
    records: List[Dict[str, Any]] = []
    cumulative_reward: float = 0.0

    def flush_chunk() -> None:
        nonlocal chunk_idx, depth_buf, labels_buf, automap_buf, screen_buf
        if depth_buf:
            np.savez_compressed(os.path.join(session_dir, f"depth_chunk_{chunk_idx:03d}.npz"), frames=np.stack(depth_buf))
            depth_buf = []
        if labels_buf:
            np.savez_compressed(os.path.join(session_dir, f"labels_chunk_{chunk_idx:03d}.npz"), frames=np.stack(labels_buf))
            labels_buf = []
        if automap_buf:
            np.savez_compressed(os.path.join(session_dir, f"automap_chunk_{chunk_idx:03d}.npz"), frames=np.stack(automap_buf))
            automap_buf = []
        if video_backend == "npz" and screen_buf:
            np.savez_compressed(os.path.join(session_dir, f"screen_chunk_{chunk_idx:03d}.npz"), frames=np.stack(screen_buf))
            screen_buf = []
        chunk_idx += 1

    try:
        while True:
            item = queue.get()
            if item is None:
                break

            itype = item.get("type", None)

            if itype == "frame":
                screen = item["screen"]
                depth  = item.get("depth", None)
                labels = item.get("labels", None)
                automap = item.get("automap", None)
                rec_row = item["rec_row"]

                if video_backend == "ffmpeg":
                    if ff_proc is None:
                        h, w = int(screen.shape[0]), int(screen.shape[1])
                        out_path = os.path.join(session_dir, f"screen.{video_container}")
                        ff_proc = _open_ffmpeg_stdin(w, h, video_fps, video_codec, video_preset, video_crf, out_path)
                    bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                    ff_proc.stdin.write(bgr.tobytes())
                elif video_backend == "opencv":
                    if vw is None:
                        h, w = int(screen.shape[0]), int(screen.shape[1])
                        vw = cv2.VideoWriter(video_path, fourcc, float(video_fps), (w, h), True)
                    bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                    vw.write(bgr)
                elif video_backend == "npz":
                    screen_buf.append(screen.astype(np.uint8, copy=False))

                if isinstance(depth, np.ndarray):
                    if depth.ndim == 3 and depth.shape[2] == 1:
                        depth = depth[:, :, 0]
                    depth_buf.append(depth.astype(np.uint16, copy=False))
                if isinstance(labels, np.ndarray):
                    if labels.ndim == 3 and labels.shape[2] == 1:
                        labels = labels[:, :, 0]
                    labels_buf.append(labels.astype(np.uint8, copy=False))
                if isinstance(automap, np.ndarray):
                    automap_buf.append(automap.astype(np.uint8, copy=False))

                max_len = max(len(depth_buf), len(labels_buf), len(automap_buf), len(screen_buf))
                if max_len >= int(chunk_size):
                    flush_chunk()

                cumulative_reward += float(rec_row.get("reward", 0.0))
                rec_row["cumulative_reward_video"] = cumulative_reward
                records.append(rec_row)

            elif itype == "finalize":
                if depth_buf or labels_buf or automap_buf or screen_buf:
                    flush_chunk()

                df = pd.DataFrame(records)
                df.to_parquet(os.path.join(session_dir, "meta.parquet"), index=False)

                meta = {
                    "button_names": item["button_names"],
                    "gamevariable_names": item["gamevariable_names"],
                    "num_steps": len(records),
                    "terminal_reason": item["terminal_reason"],
                    "cumulative_reward": cumulative_reward,
                    "doom_wad": doom_wad, "doom_map": doom_map,
                    "doom_skill": int(doom_skill),
                    "video_backend": video_backend,
                    "video_path": (
                        f"screen.{video_container}" if video_backend == "ffmpeg"
                        else ("screen.avi" if video_backend == "opencv" else "screen_chunk_XXX.npz")
                    ),
                    "video_fps": float(video_fps), "video_codec": str(video_codec),
                    "video_crf": int(video_crf), "video_preset": str(video_preset),
                    "chunk_size": int(chunk_size),
                }
                extra_meta = item.get("extra_meta", None)
                if isinstance(extra_meta, dict):
                    meta.update(extra_meta)
                with open(os.path.join(session_dir, "session_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                if ff_proc is not None:
                    try:
                        ff_proc.stdin.close()
                        ff_proc.wait()
                    except Exception:
                        pass
                if vw is not None:
                    try:
                        vw.release()
                    except Exception:
                        pass
                break

    except Exception as e:
        for res, close in [(ff_proc, lambda p: (p.stdin.close(), p.wait(2))), (vw, lambda v: v.release())]:
            if res is not None:
                try:
                    close(res)
                except Exception:
                    pass
        with open(os.path.join(session_dir, "writer_error.log"), "w", encoding="utf-8") as f:
            f.write(str(e) + "\n")
            import traceback
            f.write(traceback.format_exc())


class AsyncEpisodeRecorder:
    def __init__(
        self,
        base_dir: str,
        button_names: List[str],
        gamevar_names: List[str],
        doom_wad: str, doom_map: str, doom_skill: int,
        video_backend: str, video_container: str, video_codec: str,
        video_crf: int, video_preset: str, video_fps: float,
        chunk_size: int, queue_maxsize: int,
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"session_{ts}")
        ensure_dir(self.session_dir)
        self.button_names  = list(button_names)
        self.gamevar_names = list(gamevar_names)
        self.queue: mp.Queue = mp.Queue(maxsize=int(queue_maxsize))
        self.proc = mp.Process(
            target=_writer_process,
            args=(
                self.queue, self.session_dir,
                self.button_names, self.gamevar_names,
                doom_wad, doom_map, doom_skill,
                video_backend, video_container, video_codec,
                int(video_crf), video_preset, float(video_fps), int(chunk_size),
            ),
            daemon=True,
        )
        self.proc.start()

    def enqueue_step(
        self,
        t_index: int, obs: Dict[str, Any], action_bin: np.ndarray,
        reward: float, terminated: bool, truncated: bool,
        lives: int, reason: Optional[str], timestamp_s: Optional[float],
        doom_wad: str, doom_map: str, doom_skill: int,
    ) -> None:
        gv_dict: Dict[str, float] = {}
        if "gamevariables" in obs and isinstance(obs["gamevariables"], np.ndarray):
            for name, val in zip(self.gamevar_names, obs["gamevariables"].reshape(-1).tolist()):
                gv_dict[name] = float(val)

        rec_row = {
            "t_index": int(t_index),
            "action_bin": action_bin.astype(np.int8).tolist(),
            "action_names": self.button_names,
            "reward": float(reward),
            "is_terminal": bool(terminated), "is_timeout": bool(truncated),
            "lives": int(lives),
            "health":    gv_dict.get("HEALTH", float("nan")),
            "armor":     gv_dict.get("ARMOR",  float("nan")),
            "killcount": gv_dict.get("KILLCOUNT", float("nan")),
            "timestamp_s":   float(timestamp_s) if timestamp_s is not None else None,
            "terminal_reason": reason,
            "doom_wad": doom_wad, "doom_map": doom_map, "doom_skill": int(doom_skill),
            "selected_weapon":      gv_dict.get("SELECTED_WEAPON", float("nan")),
            "selected_weapon_ammo": gv_dict.get("SELECTED_WEAPON_AMMO", float("nan")),
            "ammo1": gv_dict.get("AMMO1", float("nan")),
            "ammo2": gv_dict.get("AMMO2", float("nan")),
            "ammo3": gv_dict.get("AMMO3", float("nan")),
            "ammo4": gv_dict.get("AMMO4", float("nan")),
            "weapon1": gv_dict.get("WEAPON1", float("nan")),
            "weapon2": gv_dict.get("WEAPON2", float("nan")),
            "weapon3": gv_dict.get("WEAPON3", float("nan")),
            "weapon4": gv_dict.get("WEAPON4", float("nan")),
            "weapon5": gv_dict.get("WEAPON5", float("nan")),
            "weapon6": gv_dict.get("WEAPON6", float("nan")),
            "weapon7": gv_dict.get("WEAPON7", float("nan")),
        }

        screen  = obs.get("screen", None)
        depth   = obs.get("depth", None)
        labels  = obs.get("labels", None)
        automap = obs.get("automap", None)
        if isinstance(depth, np.ndarray) and depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if isinstance(labels, np.ndarray) and labels.ndim == 3 and labels.shape[2] == 1:
            labels = labels[:, :, 0]

        self.queue.put({
            "type": "frame",
            "screen": screen, "depth": depth, "labels": labels, "automap": automap,
            "rec_row": rec_row,
        })

    def finalize(self, terminal_reason: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        self.queue.put({
            "type": "finalize",
            "terminal_reason": terminal_reason,
            "button_names": self.button_names,
            "gamevariable_names": self.gamevar_names,
            "extra_meta": extra_meta or {},
        })
        self.proc.join(timeout=60.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers para actualizar el HUD desde el loop
# ═══════════════════════════════════════════════════════════════════════════════

def _update_hud_state(
    hud_state: HUDState,
    obs: Dict[str, Any],
    gv_names: List[str],
    tics: int,
    fps: float,
    cumulative_reward: float,
    pressed_buttons: List[str],
) -> None:
    """Lee obs['gamevariables'] y actualiza hud_state de forma thread-safe."""
    gv_dict: Dict[str, float] = {}
    gv = obs.get("gamevariables", None)
    if isinstance(gv, np.ndarray):
        for name, val in zip(gv_names, gv.reshape(-1).tolist()):
            gv_dict[name] = float(val)

    weapons_owned = [False] * 8
    for slot in range(1, 8):
        key = f"WEAPON{slot}"
        if gv_dict.get(key, 0.0) > 0:
            weapons_owned[slot] = True

    with hud_state.lock:
        hud_state.health               = gv_dict.get("HEALTH", hud_state.health)
        hud_state.armor                = gv_dict.get("ARMOR",  hud_state.armor)
        hud_state.killcount            = int(gv_dict.get("KILLCOUNT", hud_state.killcount))
        hud_state.selected_weapon      = int(gv_dict.get("SELECTED_WEAPON", hud_state.selected_weapon))
        hud_state.selected_weapon_ammo = gv_dict.get("SELECTED_WEAPON_AMMO", hud_state.selected_weapon_ammo)
        hud_state.ammo = {
            1: gv_dict.get("AMMO1", hud_state.ammo.get(1, 0.0)),
            2: gv_dict.get("AMMO2", hud_state.ammo.get(2, 0.0)),
            3: gv_dict.get("AMMO3", hud_state.ammo.get(3, 0.0)),
            4: gv_dict.get("AMMO4", hud_state.ammo.get(4, 0.0)),
        }
        hud_state.weapons_owned    = weapons_owned
        hud_state.tics             = tics
        hud_state.fps              = fps
        hud_state.cumulative_reward = cumulative_reward
        hud_state.pressed_buttons  = pressed_buttons


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def ask_to_start() -> bool:
    try:
        ans = input("¿Deseas comenzar? [s/N]: ").strip().lower()
        return ans in ("s", "si", "sí", "y", "yes")
    except KeyboardInterrupt:
        return False


def main():
    parser = argparse.ArgumentParser(description="DOOM")
    parser.add_argument("--config",     type=str, default="game_config.yaml")
    parser.add_argument("--keymap",     type=str, default="keymap.yaml")
    parser.add_argument("--record",     action="store_true")
    parser.add_argument("--output-dir", type=str, default="recordings")
    args = parser.parse_args()

    # ── Setup pre-juego (stdout normal) ───────────────────────────────────────
    print("\n=== DOOM ===")
    if not ask_to_start():
        print("Saliendo.")
        return

    try:
        game_cfg = load_yaml(args.config)
    except Exception as e:
        print(f"ERROR al cargar config de juego: {e}")
        sys.exit(1)

    try:
        keymap = load_yaml(args.keymap)
    except Exception as e:
        print(f"ERROR al cargar keymap: {e}")
        sys.exit(1)

    scn        = game_cfg.get("scenario", {})
    doom_wad   = str(scn.get("doom_scenario_path", "unknown_wad"))
    doom_map   = str(scn.get("doom_map", "map01"))
    doom_skill = int(scn.get("doom_skill", 3))

    rec_cfg        = game_cfg.get("recording", {}) or {}
    video_backend  = str(rec_cfg.get("video_backend",  VIDEO_BACKEND_DEFAULT)).lower()
    video_container= str(rec_cfg.get("video_container",VIDEO_CONTAINER_DEFAULT))
    video_codec    = str(rec_cfg.get("video_codec",    VIDEO_CODEC_DEFAULT))
    video_crf      = int(rec_cfg.get("video_crf",      VIDEO_CRF_DEFAULT))
    video_preset   = str(rec_cfg.get("video_preset",   VIDEO_PRESET_DEFAULT))
    video_fps      = float(rec_cfg.get("video_fps",    VIDEO_FPS_DEFAULT))
    chunk_size     = int(rec_cfg.get("chunk_size",     CHUNK_SIZE_DEFAULT))
    queue_maxsize  = int(rec_cfg.get("queue_maxsize",  QUEUE_MAXSIZE_DEFAULT))

    try:
        ctrl = DoomController(config_path=args.config)
    except Exception as e:
        print(f"ERROR al crear DoomController: {e}")
        sys.exit(1)

    keys = GlobalKeyState()
    keys.start()

    obs        = ctrl.reset()
    button_names = ctrl.button_names
    gv_names   = ctrl.game_variable_names

    recorder: Optional[AsyncEpisodeRecorder] = None
    if args.record:
        ensure_dir(args.output_dir)
        recorder = AsyncEpisodeRecorder(
            base_dir=args.output_dir,
            button_names=button_names, gamevar_names=gv_names,
            doom_wad=doom_wad, doom_map=doom_map, doom_skill=doom_skill,
            video_backend=video_backend, video_container=video_container,
            video_codec=video_codec, video_crf=video_crf,
            video_preset=video_preset, video_fps=video_fps,
            chunk_size=chunk_size, queue_maxsize=queue_maxsize,
        )

    # ── Estado compartido con el HUD ──────────────────────────────────────────
    hud_state = HUDState(
        doom_wad=doom_wad, doom_map=doom_map, doom_skill=doom_skill,
        recording=args.record,
        session_dir=recorder.session_dir if recorder else "",
    )

    hud = CursesHUD(hud_state, refresh_hz=10.0)
    hud.start()

    # ── Loop principal a 35 Hz ────────────────────────────────────────────────
    target_hz  = 35.0
    period     = 1.0 / target_hz
    next_t     = time.perf_counter()
    t_start    = time.perf_counter()
    t_index    = 0
    lives      = 1
    cumulative_reward = 0.0
    terminal_reason: Optional[str] = None

    try:
        while True:
            if keys.exit_requested():
                terminal_reason = "user"
                break

            pressed      = keys.snapshot()
            action_vec   = build_action_vector(button_names, keymap, pressed)
            active_keys  = _pressed_to_button_names(button_names, keymap, pressed)

            obs, r, terminated, truncated, info = ctrl.step(action_vec, repeat=1)
            cumulative_reward += float(r)

            # ── Actualizar HUD ────────────────────────────────────────────────
            elapsed = max(1e-9, time.perf_counter() - t_start)
            fps_est = t_index / elapsed if t_index > 0 else 0.0
            _update_hud_state(
                hud_state, obs, gv_names,
                tics=t_index, fps=fps_est,
                cumulative_reward=cumulative_reward,
                pressed_buttons=active_keys,
            )

            if terminated or truncated:
                if truncated:
                    terminal_reason = "timeout"
                else:
                    try:
                        is_dead = bool(ctrl.game.is_player_dead())
                    except Exception:
                        is_dead = False
                    if is_dead:
                        terminal_reason = "death"
                    else:
                        gv = obs.get("gamevariables", None)
                        hp = None
                        if isinstance(gv, np.ndarray) and "HEALTH" in gv_names:
                            hp = float(gv[gv_names.index("HEALTH")])
                        terminal_reason = "death" if (hp is not None and hp <= 0) else "success"
                break

            if recorder is not None:
                recorder.enqueue_step(
                    t_index=t_index, obs=obs, action_bin=action_vec,
                    reward=r, terminated=terminated, truncated=truncated,
                    lives=lives, reason=None,
                    timestamp_s=(time.perf_counter() - next_t + period),
                    doom_wad=doom_wad, doom_map=doom_map, doom_skill=doom_skill,
                )

            t_index += 1
            next_t  += period
            now      = time.perf_counter()
            delay    = next_t - now
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = now

    except KeyboardInterrupt:
        terminal_reason = "user"

    finally:
        # Señalar fin al HUD y esperar que renderice la pantalla final
        with hud_state.lock:
            hud_state.running    = False
            hud_state.end_reason = terminal_reason or "user"
        hud.join(timeout=5.0)

        if recorder is not None:
            recorder.finalize(
                terminal_reason=terminal_reason or "user",
                extra_meta={
                    "total_steps": t_index,
                    "buttons": button_names,
                    "gamevariables": gv_names,
                    "config_path":  os.path.abspath(args.config),
                    "keymap_path":  os.path.abspath(args.keymap),
                    "cumulative_reward": cumulative_reward,
                    "dataset_format": {
                        "video": f"screen.{video_container} ({video_codec}, crf={video_crf}, preset={video_preset}, {video_fps} fps)",
                        "chunks": {
                            "depth": "depth_chunk_XXX.npz (uint16, (N,H,W))",
                            "labels": "labels_chunk_XXX.npz (uint8, (N,H,W))",
                            "automap": "automap_chunk_XXX.npz (uint8, (N,H,W) o (N,H,W,3))",
                            "chunk_size": chunk_size,
                        },
                        "meta": "meta.parquet (una fila por frame de video)",
                    },
                },
            )

        keys.stop()
        ctrl.close()

        # Resumen final en stdout limpio
        msg = {
            "user":    "Sesión terminada por el usuario.",
            "death":   "Perdiste! :c (muerte del jugador).",
            "success": "Ganaste! :D (Objetivo alcanzado).",
            "timeout": "Sesión terminada por límite de tiempo.",
        }.get(terminal_reason or "", "Sesión finalizada.")
        print("\n" + "=" * 60)
        print(f"Motivo de cierre : {terminal_reason}")
        print(f"Kills totales    : {hud_state.killcount}")
        print(f"Tics jugados     : {t_index}")
        print(f"Recompensa total : {cumulative_reward:+.4f}")
        print(msg)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()