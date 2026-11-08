import time
import base64
from collections import deque

import numpy as np
import cv2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from config import (
    GAME_URL, FRAME_WIDTH, FRAME_HEIGHT, FRAME_STACK, GAME_TIME_PER_STEP
)


JS_IS_CRASHED = "return Runner.instance_ ? Runner.instance_.crashed : false;"
# distanceRan * scoreCoefficient gives the actual displayed score (same as on screen)
JS_GET_SCORE  = """
    var r = Runner.instance_;
    if (!r) return 0;
    return Math.ceil(r.distanceRan * r.distanceMeter.config.COEFFICIENT);
"""
# Disable speed acceleration so difficulty stays constant — key for reaching high scores
JS_NO_ACCEL   = "Runner.config.ACCELERATION = 0;"
# Disable clouds — they are visual noise for the agent (random positions in the
# upper half of the frame) and can confuse the CNN into treating them as obstacles.
# Horizon and its config are inside the game's JS closure (not on window), so we
# can only reach them through Runner.instance_.horizon on the live object.
JS_NO_CLOUDS  = """
    var h = Runner.instance_ && Runner.instance_.horizon;
    if (h) {
        h.clouds = [];
        h.cloudFrequency = 0;
        h.config = Object.assign({}, h.config, { MAX_CLOUDS: 0, CLOUD_FREQUENCY: 0 });
        h.updateClouds = function() { this.clouds = []; };
    }
"""
JS_START_GAME = """
    var r = Runner.instance_;
    if (r.crashed) { r.restart(); }
    else if (!r.playing) { r.startGame(); }
"""
# Grab the raw canvas pixels as a base64 PNG
JS_GET_CANVAS = """
    var canvas = document.querySelector('canvas');
    return canvas.toDataURL('image/png').split(',')[1];
"""


class DinoEnv:
    def __init__(self, headless=False):
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--mute-audio")
        options.add_argument("--disable-infobars")
        options.add_argument("--window-size=900,400")
        self.driver = webdriver.Chrome(options=options)
        self.driver.get(GAME_URL)
        time.sleep(3)

        # Start the game once so Runner.instance_ exists before we patch it
        self._start_game()

        # Disable acceleration and clouds, then set our custom speed
        self.driver.execute_script(JS_NO_ACCEL)
        self.driver.execute_script(JS_NO_CLOUDS)

        # Frame stack: deque of FRAME_STACK grayscale frames
        self.frames = deque(maxlen=FRAME_STACK)

    # ------------------------------------------------------------------ #
    #  Public API                                                        #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Restart the game, return stacked initial state.
        
        The game runs in real time in the browser. After this reset, it is left
        paused so that step() controls the timing. Speed is fixed (acceleration
        disabled) and clouds are disabled to keep the observation space clean.
        """
        self.driver.execute_script(JS_START_GAME)
        time.sleep(0.5)
        
        # Capture initial frames while game is running
        frame = self._get_frame()
        for _ in range(FRAME_STACK):
            self.frames.append(frame)
        
        # Leave game paused after reset
        self.driver.execute_script("Runner.instance_.stop();")
        
        return self._stacked_state()

    def step(self, action):
        """
        Apply action, return (next_state, reward, done, score).
        Actions: 0 = nothing, 1 = jump (no duck: only cacti appear at the fixed speed).

        The game runs in real time in the browser, so it is paused with stop() between steps.
        Each step resumes it with play(), sends the action, lets it run for about
        GAME_TIME_PER_STEP seconds, pauses it again, and only then captures the frame and reads
        crashed/score. The game advances about the same amount per step no matter how long
        the agent or the learning step takes (browser timing still adds some jitter).

        play() resets the dino's status and animation to RUNNING. Without a fix, a jump in
        progress would use the slower RUNNING time scale (about four times longer in the air)
        and show running legs. So status, currentFrame, msPerFrame, currentAnimFrames and timer
        are saved before play() and restored after it.
        """
        # Check if already crashed before resuming
        crashed_before = self.driver.execute_script(JS_IS_CRASHED)
        
        if not crashed_before:
            # Resume game, preserving animation state.
            # play() calls tRex.update(0, RUNNING), which resets status, currentFrame,
            # msPerFrame, and currentAnimFrames. Save and restore them.
            self.driver.execute_script("""
                var r = Runner.instance_;
                var t = r.tRex;
                var savedStatus = t.status;
                var savedCurrentFrame = t.currentFrame;
                var savedMsPerFrame = t.msPerFrame;
                var savedCurrentAnimFrames = t.currentAnimFrames;
                var savedTimer = t.timer;
                
                r.play();
                
                // Only restore if in an active state (not WAITING or CRASHED)
                if (savedStatus === 'RUNNING' || savedStatus === 'JUMPING' || savedStatus === 'DUCKING') {
                    t.status = savedStatus;
                    t.currentFrame = savedCurrentFrame;
                    t.msPerFrame = savedMsPerFrame;
                    t.currentAnimFrames = savedCurrentAnimFrames;
                    if (savedTimer !== undefined) t.timer = savedTimer;
                }
            """)
            
            # Apply action while game is running
            self._apply_action(action)
            
            # Let game run for fixed time
            time.sleep(GAME_TIME_PER_STEP)
            
            # Pause game
            self.driver.execute_script("Runner.instance_.stop();")
        
        # Capture frame and read state while paused (or if already crashed)
        self.frames.append(self._get_frame())
        crashed = self.driver.execute_script(JS_IS_CRASHED)
        score   = self.driver.execute_script(JS_GET_SCORE)

        reward = -1.0 if crashed else 0.1
        return self._stacked_state(), reward, crashed, score

    def get_raw_frame(self):
        """Capture full-resolution canvas as RGB numpy array."""
        b64 = self.driver.execute_script(JS_GET_CANVAS)
        img_bytes = base64.b64decode(b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def close(self):
        self.driver.quit()

    # ------------------------------------------------------------------ #
    #  Internals                                                         #
    # ------------------------------------------------------------------ #

    def _start_game(self):
        body = self.driver.find_element("tag name", "body")
        body.send_keys(Keys.SPACE)
        time.sleep(1)

    def _get_frame(self):
        """Capture canvas → grayscale → 80×20 numpy array (uint8)."""
        b64 = self.driver.execute_script(JS_GET_CANVAS)
        img_bytes = base64.b64decode(b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (FRAME_WIDTH, FRAME_HEIGHT))
        return resized  # shape: (20, 80)

    def _stacked_state(self):
        """Stack frames into shape (FRAME_STACK, H, W) as float32 in [0,1]."""
        stacked = np.stack(self.frames, axis=0).astype(np.float32) / 255.0
        return stacked  # shape: (4, 20, 80)

    def _apply_action(self, action):
        """
        Apply action: 0=nothing, 1=jump
        """
        if action == 1:
            body = self.driver.find_element("tag name", "body")
            body.send_keys(Keys.SPACE)

