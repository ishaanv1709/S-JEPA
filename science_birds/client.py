"""
Science Birds Python Server — WebSocket server that the Unity game connects to.

PROTOCOL (from Assets/Scripts/AIBirdsConnection.cs):
  - Unity game is a WebSocket CLIENT connecting to ws://localhost:9000/
  - Python runs the SERVER on port 9000
  - We send commands: ["<msg_id>", "<command>", {params}]
  - Game responds:    ["<msg_id>", {data}]
  - Commands: click, drag, mousewheel, screenshot, gamestate, score, selectlevel, loadscene
"""

import json
import time
import base64
import asyncio
import threading
import numpy as np
from typing import Optional


class ScienceBirdsClient:
    """
    WebSocket SERVER that Science Birds Unity game connects to.

    Requires: pip install websockets
    The Unity game connects to ws://localhost:9000/ as a client.
    """

    STATE_MAIN_MENU = "MainMenu"
    STATE_LEVEL_SELECT = "LevelSelect"
    STATE_PLAYING = "GameWorld"
    STATE_WON = "LevelCleared"
    STATE_LOST = "LevelFailed"
    STATE_UNKNOWN = "Unknown"

    # Screen coords from actual screenshot: 918 x 399
    # Y from top (drag handler does Screen.height - y to flip)
    # Bird on slingshot position from screenshot analysis
    SLINGSHOT_X = 165.0   # Bird on slingshot X position
    SLINGSHOT_Y = 283.0   # Bird on slingshot Y position (from top)
    SCREEN_WIDTH = 918
    SCREEN_HEIGHT = 399

    def __init__(self, host: str = "localhost", port: int = 9000):
        self.host = host
        self.port = port
        self._msg_counter = 0
        self._ws = None  # The connected Unity client websocket
        self._connected = threading.Event()
        self._response = None
        self._response_event = threading.Event()
        self._loop = None
        self._server = None

    def _next_msg_id(self) -> str:
        self._msg_counter += 1
        return f"msg_{self._msg_counter}"

    def _parse_response(self, raw: str):
        """Parse response from Unity. Handles unquoted msg IDs like [msg_1,{...}]."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Unity sends [msg_1,{...}] with unquoted ID — fix it
        import re
        fixed = re.sub(r'^\[(\w+),', r'["\1",', raw)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            # Last resort: extract the JSON object part
            idx = raw.find(',')
            if idx >= 0:
                json_part = raw[idx+1:].rstrip(']')
                try:
                    return ["unknown", json.loads(json_part)]
                except json.JSONDecodeError:
                    pass
        return raw

    async def _handler(self, websocket):
        """Handle a connected Unity client."""
        print(f"  Science Birds connected!")
        self._ws = websocket
        self._connected.set()
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    message = message.decode('utf-8')
                print(f"  [RECV] {message[:200]}")
                self._response = self._parse_response(message)
                self._response_event.set()
        except Exception as e:
            print(f"  Connection error: {e}")
        finally:
            print(f"  Science Birds disconnected")
            self._ws = None
            self._connected.clear()

    async def _start_server(self):
        """Start the async WebSocket server."""
        import websockets
        self._server = await websockets.serve(
            self._handler, self.host, self.port,
            compression=None,           # Disable permessage-deflate
            max_size=10 * 1024 * 1024,  # 10MB for screenshots
            ping_interval=None,         # Don't send pings (old WebSocketSharp can't handle)
            ping_timeout=None,
        )
        await self._server.wait_closed()

    def _run_loop(self):
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())

    def connect(self, timeout: float = 30.0) -> bool:
        """Start WebSocket server and wait for the Unity game to connect."""
        try:
            import websockets
        except ImportError:
            print("ERROR: websockets not installed.")
            print("Run: pip install websockets")
            return False

        # Start server in background thread
        server_thread = threading.Thread(target=self._run_loop, daemon=True)
        server_thread.start()

        # Give server a moment to start
        time.sleep(0.5)

        print(f"  WebSocket server started on ws://{self.host}:{self.port}/")
        print(f"  Waiting for Science Birds to connect...")

        # Wait for Unity to connect
        if self._connected.wait(timeout=timeout):
            time.sleep(0.5)  # Let connection stabilize
            return True

        print(f"  No connection after {timeout}s")
        return False

    def disconnect(self):
        if self._server:
            self._server.close()
        self._ws = None

    def _send(self, command: str, params: dict = None, timeout: float = 15.0) -> dict:
        """Send command to game and wait for response. Auto-waits for reconnection."""
        if self._loop is None:
            return {}
        if params is None:
            params = {}

        # Wait for connection if disconnected
        if self._ws is None:
            print(f"  Waiting for reconnection...")
            if not self._connected.wait(timeout=10):
                print(f"  No reconnection")
                return {}
            time.sleep(0.3)

        msg_id = self._next_msg_id()
        message = json.dumps([msg_id, command, params])

        self._response = None
        self._response_event.clear()

        # Send from the asyncio loop's thread
        print(f"  [SEND] {message[:200]}")
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._ws.send(message), self._loop
            )
            future.result(timeout=5)
        except Exception as e:
            print(f"  Send error: {e}, waiting for reconnect...")
            if not self._connected.wait(timeout=10):
                return {}
            time.sleep(0.5)
            # Retry once after reconnect
            self._response = None
            self._response_event.clear()
            msg_id = self._next_msg_id()
            message = json.dumps([msg_id, command, params])
            print(f"  [RETRY] {message[:200]}")
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._ws.send(message), self._loop
                )
                future.result(timeout=5)
            except Exception:
                return {}

        # Wait for response
        if self._response_event.wait(timeout=timeout):
            resp = self._response
            if isinstance(resp, list) and len(resp) >= 2:
                return resp[1] if isinstance(resp[1], dict) else {}
            return {}
        return {}

    # ═══════════════════════════════════════
    # Game Control Commands
    # ═══════════════════════════════════════

    def get_state(self) -> str:
        response = self._send("gamestate")
        return response.get("data", self.STATE_UNKNOWN)

    def get_score(self) -> int:
        response = self._send("score")
        try:
            return int(response.get("data", "0"))
        except (ValueError, TypeError):
            return 0

    def load_level(self, level: int) -> bool:
        """Load a level. Scene reload will disconnect+reconnect WebSocket."""
        self._send("selectlevel", {"levelIndex": level})

        # Scene reload drops the connection — wait for reconnect
        self._connected.clear()
        print(f"  Waiting for game to reconnect after level load...")
        if not self._connected.wait(timeout=15):
            print(f"  Game did not reconnect after level load")
            return False

        time.sleep(2.0)  # Let level fully initialize
        state = self.get_state()
        return state == self.STATE_PLAYING

    def load_scene(self, scene: str) -> bool:
        self._send("loadscene", {"scene": scene})
        time.sleep(2.0)
        return True

    def get_screenshot(self) -> Optional[bytes]:
        response = self._send("screenshot")
        data_uri = response.get("data", "")
        if data_uri.startswith("data:image/png;base64,"):
            b64_data = data_uri.split(",", 1)[1]
            return base64.b64decode(b64_data)
        return None

    # ═══════════════════════════════════════
    # Shot Commands
    # ═══════════════════════════════════════

    def do_drag(self, start_x: float, start_y: float,
                dx: float, dy: float) -> dict:
        response = self._send("drag", {
            "x": start_x,
            "y": start_y,
            "dx": dx,
            "dy": dy,
        })
        time.sleep(5.0)
        return response

    def do_click(self, x: float, y: float) -> dict:
        return self._send("click", {"x": x, "y": y})

    def zoom(self, delta: float) -> dict:
        return self._send("mousewheel", {"delta": delta})

    def fully_zoom_out(self):
        self.zoom(-5.0)
        time.sleep(0.5)

    def shoot_bird(self, dx: float, dy: float) -> dict:
        """
        Direct bird launch — bypasses HUD raycast.
        Uses the new 'shootbird' command added to AIBirdsConnection.
        dx/dy = drag delta in screen pixels (dx>0 = pull right, dy>0 = pull down)
        For Angry Birds: pull LEFT and UP to shoot RIGHT and UP,
        so typically dx < 0 and dy > 0.
        """
        return self._send("shootbird", {"dx": dx, "dy": dy}, timeout=10.0)

    def do_shot_polar(self, angle_deg: float, power: float,
                      tap_time: float = 0.0) -> dict:
        max_pull = 70.0
        pull_distance = power * max_pull
        angle_rad = np.radians(angle_deg)

        # Pull OPPOSITE to desired launch direction
        # angle_deg=45 means launch up-right, so pull down-left
        dx = -pull_distance * np.cos(angle_rad)
        dy = pull_distance * np.sin(angle_rad)

        pre_score = self.get_score()

        # Use direct shootbird command (bypasses HUD raycast)
        result = self.shoot_bird(dx, dy)
        print(f"  [SHOOT] dx={dx:.1f}, dy={dy:.1f} -> {result}")

        # Wait for physics to settle
        time.sleep(4.0)

        if tap_time > 0:
            tap_x = self.SLINGSHOT_X + 200 * np.cos(angle_rad)
            tap_y = self.SLINGSHOT_Y - 200 * np.sin(angle_rad)
            self.do_click(tap_x, tap_y)
            time.sleep(2.0)

        post_score = self.get_score()
        state = self.get_state()

        return {
            "angle": angle_deg,
            "power": power,
            "tap_time": tap_time,
            "score_delta": post_score - pre_score,
            "state": state,
            "score": post_score,
        }

    def wait_for_stable_state(self, timeout: float = 10.0) -> str:
        start = time.time()
        while time.time() - start < timeout:
            state = self.get_state()
            if state in (self.STATE_PLAYING, self.STATE_WON, self.STATE_LOST):
                return state
            time.sleep(0.5)
        return self.STATE_UNKNOWN

    def play_level(self, level: int, actions: list) -> dict:
        self.load_level(level)
        self.fully_zoom_out()
        time.sleep(1.0)

        results = {"level": level, "shots": [], "final_score": 0, "won": False}

        for i, (angle, power, tap_time) in enumerate(actions):
            state = self.get_state()
            if state != self.STATE_PLAYING:
                if state == self.STATE_WON:
                    results["won"] = True
                break

            shot_result = self.do_shot_polar(angle, power, tap_time)
            time.sleep(2.0)

            results["shots"].append({
                "shot_index": i,
                "action": {"angle": angle, "power": power, "tap_time": tap_time},
                "score_delta": shot_result["score_delta"],
                "game_state": shot_result["state"],
            })

            if shot_result["state"] == self.STATE_WON:
                results["won"] = True
                break
            elif shot_result["state"] == self.STATE_LOST:
                break

        results["final_score"] = self.get_score()
        return results


# ═══════════════════════════════════════════════════════════
# Offline Simulator — Faithful Angry Birds physics in Python
# Used for training when Science Birds server isn't running
# ═══════════════════════════════════════════════════════════

class OfflineSimulator:
    """
    Offline physics simulator that mimics Science Birds for data collection
    when the actual game server isn't available.

    Uses 2D projectile physics with gravity, collision detection,
    and destructible structures — faithful to Angry Birds mechanics.
    """

    GRAVITY = 9.81
    MAX_LAUNCH_SPEED = 50.0
    GROUND_Y = 0.0
    BLOCK_SIZE = 30.0
    RESTITUTION = 0.3
    ARENA_WIDTH = 800
    ARENA_HEIGHT = 400
    SLINGSHOT_X = 100.0
    SLINGSHOT_Y = 200.0

    MATERIALS = {
        "wood":  {"density": 1.0, "strength": 60.0, "score": 500},
        "ice":   {"density": 0.5, "strength": 30.0, "score": 500},
        "stone": {"density": 2.0, "strength": 120.0, "score": 500},
    }

    BIRD_TYPES = {
        "red":    {"mass": 1.0, "ability": "none"},
        "blue":   {"mass": 0.5, "ability": "split"},
        "yellow": {"mass": 0.8, "ability": "speed_boost"},
        "black":  {"mass": 2.0, "ability": "explode"},
        "white":  {"mass": 1.2, "ability": "egg_bomb"},
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate_level(self, difficulty: str = "medium") -> dict:
        """Generate a random level with blocks and pigs."""
        n_blocks_map = {"easy": 6, "medium": 12, "hard": 20}
        n_pigs_map = {"easy": 2, "medium": 3, "hard": 5}
        n_birds_map = {"easy": 4, "medium": 4, "hard": 5}

        n_blocks = n_blocks_map.get(difficulty, 12)
        n_pigs = n_pigs_map.get(difficulty, 3)
        n_birds = n_birds_map.get(difficulty, 4)

        blocks = []
        base_x = self.rng.uniform(400, 600)
        materials = ["wood", "ice", "stone"]

        for i in range(n_blocks):
            col = i % 4
            row = i // 4
            material = self.rng.choice(materials, p=[0.5, 0.3, 0.2])
            blocks.append({
                "type": "rect",
                "material": material,
                "x": base_x + col * self.BLOCK_SIZE,
                "y": self.GROUND_Y + row * self.BLOCK_SIZE,
                "rotation": self.rng.choice([0.0, 90.0]) if self.rng.random() < 0.2 else 0.0,
                "health": self.MATERIALS[material]["strength"],
                "max_health": self.MATERIALS[material]["strength"],
            })

        pigs = []
        for i in range(n_pigs):
            pig_x = base_x + self.rng.uniform(0, max(1, n_blocks % 4) * self.BLOCK_SIZE)
            pig_row = min(i + 1, n_blocks // 4)
            pigs.append({
                "size": self.rng.choice(["small", "medium", "large"], p=[0.4, 0.4, 0.2]),
                "x": pig_x,
                "y": self.GROUND_Y + pig_row * self.BLOCK_SIZE + self.BLOCK_SIZE / 2,
                "health": 50.0 + self.rng.uniform(0, 50),
                "max_health": 100.0,
            })

        bird_types = list(self.BIRD_TYPES.keys())
        birds = []
        for i in range(n_birds):
            btype = bird_types[i % len(bird_types)]
            birds.append({
                "type": btype,
                "x": self.SLINGSHOT_X - 30 * i,
                "y": self.SLINGSHOT_Y,
                "used": False,
            })

        return {
            "birds": birds,
            "blocks": blocks,
            "pigs": pigs,
            "slingshot": {"x": self.SLINGSHOT_X, "y": self.SLINGSHOT_Y},
            "score": 0,
        }

    def simulate_shot(self, level: dict, bird_idx: int,
                      angle_deg: float, power: float,
                      tap_time: float = 0.0) -> dict:
        """Simulate a shot and return the resulting level state."""
        import copy
        new_level = copy.deepcopy(level)

        if bird_idx >= len(new_level["birds"]):
            return {"level": new_level, "score_delta": 0, "trajectory": [],
                    "pigs_alive": sum(1 for p in new_level["pigs"] if p["health"] > 0),
                    "birds_remaining": sum(1 for b in new_level["birds"] if not b["used"]),
                    "won": False, "lost": True}

        bird = new_level["birds"][bird_idx]
        bird["used"] = True
        bird_mass = self.BIRD_TYPES[bird["type"]]["mass"]

        speed = power * self.MAX_LAUNCH_SPEED
        angle_rad = np.radians(np.clip(angle_deg, 0, 90))
        vx = speed * np.cos(angle_rad)
        vy = speed * np.sin(angle_rad)

        px, py = self.SLINGSHOT_X, self.SLINGSHOT_Y
        dt = 0.02
        score_gained = 0
        trajectory = []

        for step in range(600):
            t = step * dt
            vy -= self.GRAVITY * dt
            px += vx * dt
            py += vy * dt
            trajectory.append((px, py))

            # Bird ability activation
            if tap_time > 0 and step == int(tap_time / dt):
                ability = self.BIRD_TYPES[bird["type"]]["ability"]
                if ability == "speed_boost":
                    vx *= 2.5
                elif ability == "explode":
                    for block in new_level["blocks"]:
                        dist = np.sqrt((block["x"] - px)**2 + (block["y"] - py)**2)
                        if dist < 80:
                            damage = bird_mass * 50 * (1 - dist / 80)
                            block["health"] = max(0, block["health"] - damage)
                            if block["health"] <= 0:
                                score_gained += self.MATERIALS[block["material"]]["score"]
                    for pig in new_level["pigs"]:
                        dist = np.sqrt((pig["x"] - px)**2 + (pig["y"] - py)**2)
                        if dist < 80:
                            pig["health"] = max(0, pig["health"] - 60 * (1 - dist / 80))
                            if pig["health"] <= 0:
                                score_gained += 5000

            # Ground collision
            if py <= self.GROUND_Y:
                py = self.GROUND_Y
                vy = -vy * self.RESTITUTION
                vx *= 0.8
                if abs(vy) < 0.5:
                    break

            # Out of bounds
            if px > self.ARENA_WIDTH or px < 0:
                break

            # Block collisions (AABB)
            for block in new_level["blocks"]:
                if block["health"] <= 0:
                    continue
                half = self.BLOCK_SIZE / 2
                if abs(px - block["x"]) < half and abs(py - block["y"]) < half:
                    impact_speed = np.sqrt(vx**2 + vy**2)
                    damage = bird_mass * impact_speed * 0.5
                    block["health"] = max(0, block["health"] - damage)
                    if block["health"] <= 0:
                        score_gained += self.MATERIALS[block["material"]]["score"]
                    vx *= -self.RESTITUTION
                    vy *= self.RESTITUTION * 0.5
                    px += vx * dt * 2
                    break

            # Pig collisions
            for pig in new_level["pigs"]:
                if pig["health"] <= 0:
                    continue
                dist = np.sqrt((px - pig["x"])**2 + (py - pig["y"])**2)
                if dist < 20:
                    impact_speed = np.sqrt(vx**2 + vy**2)
                    damage = bird_mass * impact_speed * 0.8
                    pig["health"] = max(0, pig["health"] - damage)
                    if pig["health"] <= 0:
                        score_gained += 5000
                    vx *= 0.5
                    vy *= 0.5

            speed_now = np.sqrt(vx**2 + vy**2)
            if speed_now < 0.1 and py <= self.GROUND_Y + 1:
                break

        self._apply_gravity_collapse(new_level)
        new_level["score"] += score_gained

        pigs_alive = sum(1 for p in new_level["pigs"] if p["health"] > 0)
        birds_remaining = sum(1 for b in new_level["birds"] if not b["used"])

        return {
            "level": new_level,
            "score_delta": score_gained,
            "trajectory": trajectory,
            "pigs_alive": pigs_alive,
            "birds_remaining": birds_remaining,
            "won": pigs_alive == 0,
            "lost": pigs_alive > 0 and birds_remaining == 0,
        }

    def _apply_gravity_collapse(self, level: dict):
        """Blocks above destroyed blocks fall and may cause chain damage."""
        for block in level["blocks"]:
            if block["health"] <= 0:
                continue
            for other in level["blocks"]:
                if other is block or other["health"] > 0:
                    continue
                if (abs(block["x"] - other["x"]) < self.BLOCK_SIZE and
                        block["y"] > other["y"] and
                        block["y"] - other["y"] < self.BLOCK_SIZE * 1.5):
                    block["health"] = max(0, block["health"] - 10.0)
                    block["y"] = max(self.GROUND_Y, block["y"] - self.BLOCK_SIZE)

        for pig in level["pigs"]:
            if pig["health"] <= 0:
                continue
            for block in level["blocks"]:
                if block["health"] > 0:
                    continue
                dist = np.sqrt((pig["x"] - block["x"])**2 +
                               (pig["y"] - block["y"])**2)
                if dist < self.BLOCK_SIZE:
                    pig["health"] = max(0, pig["health"] - 30)


if __name__ == "__main__":
    print("=== Testing Offline Simulator ===")
    sim = OfflineSimulator(seed=42)
    level = sim.generate_level("medium")
    print(f"Level: {len(level['birds'])} birds, "
          f"{len(level['blocks'])} blocks, {len(level['pigs'])} pigs")

    result = sim.simulate_shot(level, 0, angle_deg=35, power=0.8, tap_time=0)
    print(f"Shot: score_delta={result['score_delta']}, "
          f"pigs_alive={result['pigs_alive']}, won={result['won']}")
    print(f"Trajectory: {len(result['trajectory'])} points")

    print("\n=== Testing Science Birds Client (connection) ===")
    client = ScienceBirdsClient()
    connected = client.connect()
    if connected:
        state = client.get_state()
        print(f"Game state: {state}")
        score = client.get_score()
        print(f"Score: {score}")
        client.disconnect()
    else:
        print("Science Birds not running — use OfflineSimulator for training.")
