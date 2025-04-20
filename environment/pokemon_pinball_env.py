"""Pokemon Pinball Gymnasium environment."""
import enum
from typing import Optional, Dict, Any, Tuple

import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import Stage, BallType, SpecialMode, Maps, Pokemon

# Build a mapping enums and sequential indices
STAGE_ENUMS = list(Stage)
STAGE_TO_INDEX = {stage: idx for idx, stage in enumerate(STAGE_ENUMS)}
INDEX_TO_STAGE = {idx: stage for idx, stage in enumerate(STAGE_ENUMS)}

BALL_TYPE_ENUMS = list(BallType)
BALL_TYPE_TO_INDEX = {ball_type: idx for idx, ball_type in enumerate(BALL_TYPE_ENUMS)}
INDEX_TO_BALL_TYPE = {idx: ball_type for idx, ball_type in enumerate(BALL_TYPE_ENUMS)}


class Actions(enum.Enum):
    IDLE = 0
    LEFT_FLIPPER_PRESS = 1
    RIGHT_FLIPPER_PRESS = 2
    LEFT_FLIPPER_RELEASE = 3
    RIGHT_FLIPPER_RELEASE = 4
    LEFT_TILT = 5
    RIGHT_TILT = 6
    UP_TILT = 7
    LEFT_UP_TILT = 8
    RIGHT_UP_TILT = 9


# Global observation space dimensions
OBSERVATION_SHAPE = (16, 20)
DEFAULT_OBSERVATION_SPACE = spaces.Box(low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8)

DEFAULT_CONFIG = {
    "debug": False,
    "headless": False,
    "reward_shaping": None,
    "info_level": 2,
    "frame_stack": 4,
    "frame_skip": 3,
    "visual_mode": "game_area",  # alternative is "screen" for full RGB screen
    "frame_stack_extra_observation": False,
    "reduce_screen_resolution": True,  # Downsample full screen by factor of 2 when using "screen" mode
}


class PokemonPinballEnv(gym.Env):
    """Pokemon Pinball environment for reinforcement learning."""
    
    metadata = {"render_modes": ["human"]}
    
    # Class variable to keep track of the number of environment instances
    instance_count = 0
    
    def __init__(self, rom_path="pokemon_pinball.gbc", config=DEFAULT_CONFIG ): 
        """
        Initialize the Pokemon Pinball environment.
        
        Args:
            pyboy: PyBoy instance
            debug: Enable debug mode with normal speed for visualization
            headless: Run without visualization at maximum speed
            reward_shaping: Optional custom reward shaping function
            info_level: Level of detail in info dict (0-3, higher=more info but slower)
        """
        super().__init__()
        
        # Increment the instance count when a new environment is created
        PokemonPinballEnv.instance_count += 1
        # Store the instance ID
        self.instance_id = PokemonPinballEnv.instance_count
        
        import os
        pid = os.getpid()
        
        # Only try to detect driver environment in the first instance per process
        # We'll limit this to just the main thread in the main process
        if self.instance_id == 1:
            import traceback
            
            # Look at the call stack - if this is being created by PufferLib vector
            # initialization code for driver/observation space purposes, make it headless
            is_driver = False
            
            # Check the call stack for driver environment patterns
            stack = traceback.extract_stack()
            for frame in stack:
                if any(x in frame.filename for x in ['pufferlib/vector.py', 'pufferlib/environment.py']):
                    if any(x in frame.name for x in ['make', '_create_driver', '_reset_env']):
                        is_driver = True
                        break
            
            if is_driver:
                print(f"Detected driver environment in PID {pid} - forcing headless mode")
                config['headless'] = True
                
        print(f"Creating PokemonPinballEnv instance {self.instance_id} in process {pid}")
        window_type = "null" if config['headless'] else "SDL2"
        self.pyboy = PyBoy(rom_path, window=window_type, sound_emulated=False)
        if self.pyboy is None:
            raise ValueError("PyBoy instance is required")
        assert self.pyboy.cartridge_title == "POKEPINBALLVPH", "Invalid ROM: Pokémon Pinball required"
        
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0  # Track frames played in current episode
        
        # Episode tracking
        self._high_score = 0  # Track highest score seen
        self._episode_count = 0
        
        # Initialize tracking variables used in reward shaping
        self._prev_caught = 0
        self._prev_evolutions = 0
        self._prev_stages_completed = 0  
        self._prev_ball_upgrades = 0
        
        self.debug = config['debug']
        self.headless = config['headless']

        self.frame_skip = config['frame_skip']
        self.frame_stack = config['frame_stack']
        self.frame_stack_extra_observation = config['frame_stack_extra_observation']
        self.visual_mode = config['visual_mode']
        self.reduce_screen_resolution = config.get('reduce_screen_resolution', True)
        
        # Stuck detection parameters (always enabled)
        self.stuck_detection_window = max(50, self.frame_stack * 5)  # At least 5x the frame stack
        self.stuck_detection_threshold = 5.0  # Movement threshold in pixels
        self.stuck_detection_reward_threshold = 100  # Min score change to avoid being "stuck"
        
        # Buffer for tracking ball positions
        self.ball_position_history = []
        self.last_score = 0

        self.episodes_completed = 0

        # Set output shape based on visual mode
        if self.visual_mode == "game_area":
            self.output_shape = (16, 20, self.frame_stack)
        else:  # screen mode
            if self.reduce_screen_resolution:
                self.output_shape = (72, 80, self.frame_stack)  # Downsampled by factor of 2
            else:
                self.output_shape = (144, 160, self.frame_stack)

        # Initialize frame stacking structures if needed
        if self.frame_stack_extra_observation:
            # Create arrays to store recent ball positions and velocities for frame stacking
            self.recent_ball_x = np.zeros((self.frame_stack,), dtype=np.float32)
            self.recent_ball_y = np.zeros((self.frame_stack,), dtype=np.float32)
            self.recent_ball_x_velocity = np.zeros((self.frame_stack,), dtype=np.float32)
            self.recent_ball_y_velocity = np.zeros((self.frame_stack,), dtype=np.float32)
        
        # Configure speed based on mode
        if self.debug:
            # Normal speed for debugging
            self.pyboy.set_emulation_speed(1.0)
        else:
            # Maximum speed (0 = no limit)
            self.pyboy.set_emulation_speed(0)
            
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = self._create_observation_space(config['info_level'])
        
        # Set the reward shaping function
        reward_shaping_name = config['reward_shaping']
        if reward_shaping_name == 'basic':
            self.reward_shaping = RewardShaping.basic
        elif reward_shaping_name == 'catch_focused':
            self.reward_shaping = RewardShaping.catch_focused
        elif reward_shaping_name == 'comprehensive':
            self.reward_shaping = RewardShaping.comprehensive
        else:
            self.reward_shaping = None
            
        self.info_level = config['info_level']
        
        self._game_wrapper = self.pyboy.game_wrapper
        
        # Initialize game
        self._game_wrapper.start_game()

    def _create_observation_space(self, info_level):
        """Create an observation space based on the info level."""
        # Base space is always the visual observation (game_area or screen)
        observations_dict = {}
        
        # Create the appropriate observation space based on visual mode
        if self.visual_mode == "game_area":
            # Game area is a 2D grid with stacked frames (H, W, frames)
            observations_dict.update({
                'game_area': spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)
            })
        else:
            # Screen is RGB, so shape is (H, W, 3*frames) or (H, W, frames) if grayscale
            if len(self.output_shape) == 3:
                # Already includes channel dimension
                observations_dict.update({
                    'game_area': spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)
                })
            else:
                # Add RGB channels (3) for each frame
                h, w = self.output_shape[:2]
                screen_shape = (h, w, 3 * self.frame_stack)
                observations_dict.update({
                    'game_area': spaces.Box(low=0, high=255, shape=screen_shape, dtype=np.uint8)
                })
        
        # Add spaces based on info level
        if info_level >= 1:
            # Level 1 - ball position and velocity
            obs_shape=(1,)
            if self.frame_stack_extra_observation:
                obs_shape=(1,self.frame_stack)

            observations_dict.update({
                'ball_x': spaces.Box(low=-128, high=128, shape=obs_shape, dtype=np.float32),
                'ball_y': spaces.Box(low=-128, high=128, shape=obs_shape,dtype=np.float32),
                'ball_x_velocity': spaces.Box(low=-128, high=128, shape=obs_shape, dtype=np.float32),
                'ball_y_velocity': spaces.Box(low=-128, high=128, shape=obs_shape, dtype=np.float32),
            })
        
        if info_level >= 2:
            # Level 2 - Additional game state

            observations_dict.update({
                'current_stage': spaces.Discrete(len(STAGE_ENUMS)),
                'ball_type': spaces.Discrete(len(BALL_TYPE_ENUMS)),
                'special_mode': spaces.Discrete(len(SpecialMode)),
                'special_mode_active': spaces.Discrete(2),  # Boolean (0/1)
                'saver_active': spaces.Discrete(2),    # Boolean (0/1)
            })
            
        if info_level >= 3:
            # Level 3 - Most detailed information
            observations_dict.update({
                'pikachu_saver_charge': spaces.Discrete(16), # Pikachu saver charge values go from 0-15
            })
        
        return spaces.Dict(observations_dict)
        
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        
        action_release_delay = math.ceil((1 + self.frame_skip) / 2)

        action_map = {
            Actions.LEFT_FLIPPER_PRESS.value: lambda: self.pyboy.button_press("left"),
            Actions.RIGHT_FLIPPER_PRESS.value: lambda: self.pyboy.button_press("a"),
            Actions.LEFT_FLIPPER_RELEASE.value: lambda: self.pyboy.button_release("left"),
            Actions.RIGHT_FLIPPER_RELEASE.value: lambda: self.pyboy.button_release("a"),
            Actions.LEFT_TILT.value: lambda: self.pyboy.button("down",action_release_delay),
            Actions.RIGHT_TILT.value: lambda: self.pyboy.button("b",action_release_delay),
            Actions.UP_TILT.value: lambda: self.pyboy.button("select",action_release_delay),
            Actions.LEFT_UP_TILT.value: lambda: (self.pyboy.button("select",action_release_delay), self.pyboy.button("down",action_release_delay)),
            Actions.RIGHT_UP_TILT.value: lambda: (self.pyboy.button("select",action_release_delay), self.pyboy.button("b",action_release_delay)),
        }
        
        # Execute the action if it's not IDLE
        if action > 0 and action < len(Actions):
            action_func = action_map.get(action)
            if action_func:
                action_func()
            
        # Perform the game tick
        ticks = 1 + self.frame_skip
        self.pyboy.tick(ticks, not self.headless, False)
        self._frames_played += ticks

        # Get game state
        self._calculate_fitness()
        
        # Determine if game is over
        #done = self._game_wrapper.game_over
        done = self._game_wrapper.lost_ball_during_saver or self._game_wrapper.game_over or self._game_wrapper.balls_left<2
        
        # Apply reward shaping by calling the reward function
        # but passing self for tracking state
        if self.reward_shaping:
            reward = self._apply_reward_shaping(self.reward_shaping)
        else:
            reward = self._fitness - self._previous_fitness
            
        # Get observation
        observation = self._get_obs()
        
        # Check for stuck ball (if enabled)
        truncated = self._check_if_stuck(observation)
        
        # Get info with appropriate level of detail
        info = self._get_info()
        
        # Add episode completion flag when episode ends
        if done or truncated:
            info['episode_complete'] = [True]
            info['episode_length'] = [float(self._frames_played)]
            
            # Add high score flag if appropriate (for tracking best episodes)
            if self._game_wrapper.score > self._high_score:
                self._high_score = self._game_wrapper.score
                info['high_score'] = [True]
        
        # Add stuck status to info if applicable
        if truncated:
            info['truncated_reason'] = ['stuck_ball']
            self.reset()
        
        return observation, reward, done, truncated, info
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset reward tracking variables (instance variables)
        self._prev_caught = 0
        self._prev_evolutions = 0
        self._prev_stages_completed = 0
        self._prev_ball_upgrades = 0
        
        game_wrapper = self._game_wrapper
        game_wrapper.reset_game()
        # this method currently is not in the official pyboy API, but there is a pull request to add it
        game_wrapper.reset_tracking()
        
        # Reset fitness tracking
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0  # Reset frame counter
        
        # Reset stuck detection
        self.ball_position_history = []
        self.last_score = 0
        
        # Clear frame buffer to ensure fresh initialization
        buffer_key = f"{self.visual_mode}_buffer"
        if hasattr(self, buffer_key):
            delattr(self, buffer_key)
        
        # Reset frame stacking for ball positions if used
        if self.frame_stack_extra_observation:
            self.recent_ball_x = np.zeros((self.frame_stack,), dtype=np.float32)
            self.recent_ball_y = np.zeros((self.frame_stack,), dtype=np.float32)
            self.recent_ball_x_velocity = np.zeros((self.frame_stack,), dtype=np.float32)
            self.recent_ball_y_velocity = np.zeros((self.frame_stack,), dtype=np.float32)
        
        # Track episode completion
        self.episodes_completed += 1
        self._episode_count += 1

        # Get observation and info once
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
        
    def render(self, mode="human"):
        """Render the environment. PyBoy handles this internally."""
        pass
        
    def close(self):
        """Close the environment and stop PyBoy."""
        self.pyboy.stop()
        # Decrement the instance count when an environment is closed
        PokemonPinballEnv.instance_count = max(0, PokemonPinballEnv.instance_count - 1)
        
    @classmethod
    def get_instance_count(cls):
        """Get the current number of environment instances."""
        return cls.instance_count
        
    def _get_info(self):
        """Get additional information from the environment."""
        game_wrapper = self._game_wrapper
        
        # Convert all numeric values to lists for PufferLib compatibility
        # Basic info (always included)
        info = {
            'score': [float(game_wrapper.score)],
            'episode_return': [float(self._fitness)],
            'episode_length': [float(self._frames_played)],
            'agent_episodes_completed': [float(self.episodes_completed)],
            'episode_id': [float(self._episode_count)],
            'episode_complete': [False],  # Will be set to True when episode ends
        }
        
        # Game progress info
        info.update({
            'pokemon_caught': [float(game_wrapper.pokemon_caught_in_session)],
            'evolutions': [float(game_wrapper.evolution_success_count)],
            'ball_saver_active': [float(game_wrapper.ball_saver_seconds_left > 0)],
            'current_stage': [str(game_wrapper.current_stage)],
            'ball_type': [str(game_wrapper.ball_type)],
            'special_mode_active': [float(game_wrapper.special_mode_active)],
            'pikachu_saver_charge': [float(game_wrapper.pikachu_saver_charge)]
        })
        
        # Stage completion info - use lists for PufferLib compatibility
        total_stages = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        
        info.update({
            'diglett_stages': [float(game_wrapper.diglett_stages_completed)],
            'gengar_stages': [float(game_wrapper.gengar_stages_completed)],
            'meowth_stages': [float(game_wrapper.meowth_stages_completed)],
            'seel_stages': [float(game_wrapper.seel_stages_completed)],
            'mewtwo_stages': [float(game_wrapper.mewtwo_stages_completed)],
            'total_stages_completed': [float(total_stages)]
        })
        
        # Ball upgrade info - use lists for PufferLib compatibility
        total_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        
        info.update({
            'great_ball_upgrades': [float(game_wrapper.great_ball_upgrades)],
            'ultra_ball_upgrades': [float(game_wrapper.ultra_ball_upgrades)],
            'master_ball_upgrades': [float(game_wrapper.master_ball_upgrades)],
            'total_ball_upgrades': [float(total_upgrades)]
        })
        
        # Ball position and velocity (useful for analysis)
        info.update({
            'ball_x': [float(game_wrapper.ball_x)],
            'ball_y': [float(game_wrapper.ball_y)],
            'ball_x_velocity': [float(game_wrapper.ball_x_velocity)],
            'ball_y_velocity': [float(game_wrapper.ball_y_velocity)]
        })
        
        return info
        
    def _get_obs(self):
        """
        Get the current observation from the environment.
        """
        game_wrapper = self._game_wrapper
        
        # Get observation based on visual mode
        if self.visual_mode == "game_area":
            # Use game_area (simplified 16x20 grid)
            visual_obs = game_wrapper.game_area()
        else:
            # Get full screen image (RGB)
            screen_img = np.array(self.pyboy.screen_image())
            
            # Convert RGB to grayscale for simplicity (optional)
            # screen_img = np.mean(screen_img, axis=2, keepdims=False).astype(np.uint8)
            
            # Downsample if needed
            if self.reduce_screen_resolution:
                # Downsample by factor of 2
                h, w = screen_img.shape[:2]
                screen_img = screen_img[::2, ::2]
            
            visual_obs = screen_img
        
        # Apply frame stacking if enabled
        if self.frame_stack > 1:
            buffer_key = f"{self.visual_mode}_buffer"
            
            # For the first observation, duplicate the single frame to all frames
            if not hasattr(self, buffer_key):
                # Create a buffer with the right dimensions
                if self.visual_mode == "game_area":
                    # Add channel dimension for stacking
                    frame_buffer = np.repeat(visual_obs[:, :, np.newaxis], self.frame_stack, axis=2)
                else:
                    # For RGB screen, it already has a channel dimension
                    # Shape is (H, W, 3) -> create (H, W, 3*frame_stack)
                    if len(visual_obs.shape) == 3:  # RGB image
                        h, w, c = visual_obs.shape
                        frame_buffer = np.zeros((h, w, self.frame_stack, c), dtype=np.uint8)
                        for i in range(self.frame_stack):
                            frame_buffer[:, :, i, :] = visual_obs
                        # Reshape to (H, W, frame_stack*C)
                        frame_buffer = frame_buffer.reshape(h, w, self.frame_stack * c)
                    else:
                        # Grayscale
                        frame_buffer = np.repeat(visual_obs[:, :, np.newaxis], self.frame_stack, axis=2)
                
                setattr(self, buffer_key, frame_buffer)
            else:
                # Get the existing buffer
                frame_buffer = getattr(self, buffer_key)
                
                # Update buffer based on format
                if self.visual_mode == "game_area" or len(visual_obs.shape) == 2:
                    # For game_area or grayscale, simple rollover
                    frame_buffer = np.roll(frame_buffer, shift=-1, axis=2)
                    frame_buffer[:, :, -1] = visual_obs
                else:
                    # For RGB, we need to handle multiple channels
                    h, w, c = visual_obs.shape
                    # Get shape of buffer
                    buffer_h, buffer_w, buffer_c = frame_buffer.shape
                    # Reshape to (H, W, frame_stack, C)
                    temp_buffer = frame_buffer.reshape(h, w, self.frame_stack, c)
                    # Roll along frame_stack dimension
                    temp_buffer = np.roll(temp_buffer, shift=-1, axis=2)
                    # Update newest frame
                    temp_buffer[:, :, -1, :] = visual_obs
                    # Reshape back
                    frame_buffer = temp_buffer.reshape(h, w, self.frame_stack * c)
                
                # Save back to object
                setattr(self, buffer_key, frame_buffer)
            
            # Use the frame buffer as observation
            visual_obs = getattr(self, buffer_key)
        
        # Create observation dictionary
        observation = {
            "game_area": visual_obs,
        }
        # Level 0 - no info
        if self.info_level == 0:
            return observation
        
        # Level 1 - position and velocity information
        if self.frame_stack_extra_observation:
            # Update the recent positions and velocities
            self.recent_ball_x = np.roll(self.recent_ball_x, shift=-1)
            self.recent_ball_y = np.roll(self.recent_ball_y, shift=-1)
            self.recent_ball_x_velocity = np.roll(self.recent_ball_x_velocity, shift=-1)
            self.recent_ball_y_velocity = np.roll(self.recent_ball_y_velocity, shift=-1)
            
            # Add new values
            self.recent_ball_x[-1] = game_wrapper.ball_x
            self.recent_ball_y[-1] = game_wrapper.ball_y
            self.recent_ball_x_velocity[-1] = game_wrapper.ball_x_velocity
            self.recent_ball_y_velocity[-1] = game_wrapper.ball_y_velocity
            
            observation.update({
                "ball_x": self.recent_ball_x,
                "ball_y": self.recent_ball_y,
                "ball_x_velocity": self.recent_ball_x_velocity,
                "ball_y_velocity": self.recent_ball_y_velocity,
            })
        else:
            observation.update({
                "ball_x": game_wrapper.ball_x,
                "ball_y": game_wrapper.ball_y,
                "ball_x_velocity": game_wrapper.ball_x_velocity,
                "ball_y_velocity": game_wrapper.ball_y_velocity,
            })
        
        if self.info_level == 1:
            return observation
        
        # Level 2 - More detailed information
        if self.info_level >= 2:
            observation.update({
                "current_stage": STAGE_TO_INDEX.get(game_wrapper.current_stage),
                "ball_type": BALL_TYPE_TO_INDEX.get(game_wrapper.ball_type),
                "special_mode": game_wrapper.special_mode,
                "special_mode_active": game_wrapper.special_mode_active,
            })
            
        if game_wrapper.ball_saver_seconds_left > 0:
            observation["saver_active"] = True
        else :
            observation["saver_active"] = False
            # Level 3 - Most detailed information
        if self.info_level >= 3:
            observation["pikachu_saver_charge"] = game_wrapper.pikachu_saver_charge
            # TODO add the following
            # current map
            #

        
        return observation
        
    def _calculate_fitness(self):
        """Calculate fitness based on the game score."""
        self._previous_fitness = self._fitness
        self._fitness = self._game_wrapper.score
        
    def _apply_reward_shaping(self, reward_function):
        """Apply the reward shaping function, but handle tracking inside the environment.
        
        This ensures each environment instance maintains its own tracking state.
        """
        # Create a state dictionary to pass to the reward functions
        # This captures the current state of the reward trackers
        reward_state = {
            'prev_caught': self._prev_caught,
            'prev_evolutions': self._prev_evolutions,
            'prev_stages_completed': self._prev_stages_completed,
            'prev_ball_upgrades': self._prev_ball_upgrades
        }
        
        # Call the appropriate reward function with instance state
        reward = reward_function(
            self._fitness, 
            self._previous_fitness, 
            self._game_wrapper, 
            self._frames_played,
            reward_state
        )
        
        # Update the instance variables based on current game state
        # This ensures each environment instance maintains its own tracking variables
        self._prev_caught = self._game_wrapper.pokemon_caught_in_session
        self._prev_evolutions = self._game_wrapper.evolution_success_count
        
        # Calculate total stages completed
        total_stages_completed = (
            self._game_wrapper.diglett_stages_completed +
            self._game_wrapper.gengar_stages_completed +
            self._game_wrapper.meowth_stages_completed +
            self._game_wrapper.seel_stages_completed +
            self._game_wrapper.mewtwo_stages_completed
        )
        self._prev_stages_completed = total_stages_completed
        
        # Calculate total ball upgrades
        ball_upgrades = (
            self._game_wrapper.great_ball_upgrades +
            self._game_wrapper.ultra_ball_upgrades +
            self._game_wrapper.master_ball_upgrades
        )
        self._prev_ball_upgrades = ball_upgrades
        
        return reward
        
    def _check_if_stuck(self, observation):
        """
        Check if the ball is stuck in the same area for too long.
        
        Args:
            observation: Current observation
            
        Returns:
            Boolean indicating if episode should be truncated due to stuck ball
        """
        # Get current ball position
        game_wrapper = self._game_wrapper
        current_x = game_wrapper.ball_x
        current_y = game_wrapper.ball_y
        current_score = game_wrapper.score
        
        # Add to history
        self.ball_position_history.append((current_x, current_y))
        
        # Only check after we have enough history
        if len(self.ball_position_history) < self.stuck_detection_window:
            return False
            
        # Keep history at fixed size
        if len(self.ball_position_history) > self.stuck_detection_window:
            self.ball_position_history.pop(0)
        
        # Check if score has changed enough (if score increases a lot, agent isn't stuck)
        score_change = current_score - self.last_score
        if score_change > self.stuck_detection_reward_threshold:
            # Reset history if significant progress was made
            self.ball_position_history = []
            self.last_score = current_score
            return False
            
        # Calculate maximum distance moved in any direction
        max_x_change = 0
        max_y_change = 0
        for x, y in self.ball_position_history:
            x_change = abs(current_x - x)
            y_change = abs(current_y - y)
            max_x_change = max(max_x_change, x_change)
            max_y_change = max(max_y_change, y_change)
            
        # If both x and y movement are below threshold, ball is stuck
        if max_x_change < self.stuck_detection_threshold and max_y_change < self.stuck_detection_threshold:
            return True
            
        # Update last score
        self.last_score = current_score
        return False


class RewardShaping:
    """
    Collection of reward shaping functions for Pokemon Pinball.
    These methods operate on the environment instance to ensure proper per-instance state tracking.
    """
    
    @staticmethod
    def basic(current_fitness, previous_fitness, game_wrapper, frames_played=0, prev_state=None):
        """Basic reward shaping based on score difference."""
        return current_fitness - previous_fitness
    
    @staticmethod
    def catch_focused(current_fitness, previous_fitness, game_wrapper, frames_played=0, prev_state=None):
        """
        Reward focused on catching Pokemon.
        
        Args:
            current_fitness: Current score
            previous_fitness: Previous score
            game_wrapper: Game wrapper instance
            frames_played: Number of frames played
            prev_state: Dictionary containing previous state values for tracking
        """
        # If no state is passed, create empty state (should never happen since state is managed by environment)
        if prev_state is None:
            prev_state = {'prev_caught': 0}
            
        score_reward = (current_fitness - previous_fitness) * 0.5
        
        # Big reward for catching Pokemon
        catch_reward = 0
        if game_wrapper.pokemon_caught_in_session > prev_state.get('prev_caught', 0):
            catch_reward = 1000
            # Note: we don't modify prev_state here, as the caller is responsible for tracking it
            
        return score_reward + catch_reward
    
    @staticmethod
    def comprehensive(current_fitness, previous_fitness, game_wrapper, frames_played=0, prev_state=None):
        """
        Comprehensive reward that promotes long survival and steady progress.
        
        Args:
            current_fitness: Current score
            previous_fitness: Previous score
            game_wrapper: Game wrapper instance
            frames_played: Number of frames played
            prev_state: Dictionary containing previous state values for tracking
        """
        # If no state is passed, create empty state (should never happen since state is managed by environment)
        if prev_state is None:
            prev_state = {
                'prev_caught': 0,
                'prev_evolutions': 0, 
                'prev_stages_completed': 0,
                'prev_ball_upgrades': 0
            }
            
        # Log-scaled score difference
        score_diff = current_fitness - previous_fitness
        if score_diff > 0:
            import numpy as np
            score_reward = 15 * np.log(1 + score_diff / 100)
        else:
            score_reward = 0

        # Ball alive reward and survival bonus
        ball_alive_reward = 25
        time_bonus = min(120, frames_played / 400)

        additional_reward = 0
        reward_sources = {}

        # Catching Pokémon
        prev_caught = prev_state.get('prev_caught', 0)
        if game_wrapper.pokemon_caught_in_session > prev_caught:
            pokemon_reward = 500
            additional_reward += pokemon_reward
            reward_sources["pokemon_catch"] = pokemon_reward
            # Note: we don't modify prev_state here, as the caller is responsible for tracking it

        # Evolution rewards
        prev_evolutions = prev_state.get('prev_evolutions', 0)
        if game_wrapper.evolution_success_count > prev_evolutions:
            evolution_reward = 1000
            additional_reward += evolution_reward
            reward_sources["evolution"] = evolution_reward
            # State is updated by the environment

        # Stage completion
        total_stages_completed = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        prev_stages_completed = prev_state.get('prev_stages_completed', 0)
        if total_stages_completed > prev_stages_completed:
            stage_reward = 1500
            additional_reward += stage_reward
            reward_sources["stage_completion"] = stage_reward
            # State is updated by the environment

        # Ball upgrades
        ball_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        prev_ball_upgrades = prev_state.get('prev_ball_upgrades', 0)
        if ball_upgrades > prev_ball_upgrades:
            upgrade_reward = 200
            additional_reward += upgrade_reward
            reward_sources["ball_upgrade"] = upgrade_reward
            # State is updated by the environment

        # Combine all rewards
        total_reward = score_reward + additional_reward + ball_alive_reward + time_bonus

        return total_reward