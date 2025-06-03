"""Pokemon Pinball Gymnasium environment."""
import enum
from typing import Optional, Dict, Any, Tuple

import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces
from pyboy import PyBoy
from pyboy.plugins.game_wrapper_pokemon_pinball import Stage, BallType, SpecialMode, Maps, Pokemon
from rewards import RewardShaping

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
    #LEFT_TILT = 5
    #RIGHT_TILT = 6
    #UP_TILT = 7
    #LEFT_UP_TILT = 8
    #RIGHT_UP_TILT = 9


# Global observation space dimensions
OBSERVATION_SHAPE = (16, 20)
DEFAULT_OBSERVATION_SPACE = spaces.Box(low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8)

DEFAULT_CONFIG = {
    "debug": False,
    "headless": False,
    "reward_shaping": "comprehensive", 
    "info_level": 1,
    "frame_skip": 4,
    "visual_mode": "screen",  # alternative is "screen" for full RGB screen
    "reduce_screen_resolution": True,  # Downsample full screen by factor of 2 when using "screen" mode
    "max_episode_frames": 0, # 0 means no limit
    "episode_mode":"life", # life, ball, or game
    "reset_condition":"game", # life, ball, or game
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
            rom_path: Path to the Pokemon Pinball ROM file
            config: Configuration dictionary with various settings
        """
        super().__init__()
        
        # Increment the instance count when a new environment is created
        PokemonPinballEnv.instance_count += 1
        # Store the instance ID
        self.instance_id = PokemonPinballEnv.instance_count
        
        import os
        pid = os.getpid()
        
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

        self.frame_skip = max(1,config['frame_skip'])
        self.visual_mode = config['visual_mode']
        self.reduce_screen_resolution = config.get('reduce_screen_resolution', True)


        # Episode mode and reset condition configuration
        self.episode_mode = config.get('episode_mode', 'life')  # life, ball, or game
        self.reset_condition = config.get('reset_condition', 'game')  # life, ball, or game
        
        # Variables for tracking game state between episodes
        self._previous_balls_left = 2  # Default starting balls
        self._previous_balls_lost_during_saver = 0 
        self._initialized = False  
        
        # Stuck detection parameters 
        self.stuck_detection_window = 100
        self.stuck_detection_threshold = 5.0  
        self.stuck_detection_reward_threshold = 100  # Min score change to avoid being "stuck"
        
        # Buffer for tracking ball positions
        self.ball_position_history = []
        self.last_score = 0

        self.episodes_completed = 0

        # Set output shape based on visual mode
        if self.visual_mode == "game_area":
            # Game area in Pokemon Pinball is a 16x20 grid
            self.output_shape = (16, 20)
        else:  # screen mode
            if self.reduce_screen_resolution:
                self.output_shape = (72, 80)  # Downsampled by factor of 2
            else:
                self.output_shape = (144, 160)

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
        elif reward_shaping_name == 'progressive':
            self.reward_shaping = RewardShaping.progressive
        else:
            self.reward_shaping = None
            
        self.info_level = config['info_level']
        
        self._game_wrapper = self.pyboy.game_wrapper
        
        # Initialize game
        self._game_wrapper.start_game()

        if self.debug:
            # Normal speed for debugging
            self.pyboy.set_emulation_speed(1.0)
        else:
            # Maximum speed (0 = no limit)
            self.pyboy.set_emulation_speed(0)

    def _create_observation_space(self, info_level):
        """Create an observation space based on the info level."""
        # Base space is always the visual observation (game_area or screen)
        observations_dict = {}
        
        # Create the appropriate observation space based on visual mode
        if self.visual_mode == "game_area":
            # Game area is a 2D grid (16x20)
            observations_dict.update({
                'visual_representation': spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)
            })
        else:
            # Screen is grayscale (either 144x160xframes or 72x80xframes if downsampled)
            observations_dict.update({
                'visual_representation': spaces.Box(low=0, high=255, shape=self.output_shape, dtype=np.uint8)
            })

        if info_level == 0:
            return observations_dict.get('visual_representation')
        
        # Add spaces based on info level
        if info_level >= 1:
            # Level 1 - ball position and velocity
            obs_shape=(1,)

            observations_dict.update({
                'ball_x': spaces.Box(low=-128, high=128, shape=obs_shape , dtype=np.float32),
                'ball_y': spaces.Box(low=-128, high=128, shape=obs_shape , dtype=np.float32),
                'ball_x_velocity': spaces.Box(low=-128, high=128, shape=obs_shape , dtype=np.float32),
                'ball_y_velocity': spaces.Box(low=-128, high=128, shape=obs_shape , dtype=np.float32),
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
    
    @property
    def ball_lost(self):
        """
        Determines if a ball was just lost based on current and previous state.
        This is different from life_lost because it includes balls lost during saver
        """
        lives_decreased = (self._game_wrapper.balls_left < self._previous_balls_left)
        balls_lost_during_saver_increased = (self._game_wrapper.lost_ball_during_saver > self._previous_balls_lost_during_saver)
        
        return lives_decreased or balls_lost_during_saver_increased

    @property
    def life_lost(self):
        lives_decreased = (self._game_wrapper.balls_left < self._previous_balls_left)
        return lives_decreased

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"
        
        # Save current state before taking action
        self._previous_balls_left = self._game_wrapper.balls_left
        
        action_release_delay = math.ceil((1 + self.frame_skip) / 2)

        action_map = {
            Actions.LEFT_FLIPPER_PRESS.value: lambda: self.pyboy.button_press("left"),
            Actions.RIGHT_FLIPPER_PRESS.value: lambda: self.pyboy.button_press("a"),
            Actions.LEFT_FLIPPER_RELEASE.value: lambda: self.pyboy.button_release("left"),
            Actions.RIGHT_FLIPPER_RELEASE.value: lambda: self.pyboy.button_release("a"),
            #Actions.LEFT_TILT.value: lambda: self.pyboy.button("down",action_release_delay),
            #Actions.RIGHT_TILT.value: lambda: self.pyboy.button("b",action_release_delay),
            #Actions.UP_TILT.value: lambda: self.pyboy.button("select",action_release_delay),
            #Actions.LEFT_UP_TILT.value: lambda: (self.pyboy.button("select",action_release_delay), self.pyboy.button("down",action_release_delay)),
            #Actions.RIGHT_UP_TILT.value: lambda: (self.pyboy.button("select",action_release_delay), self.pyboy.button("b",action_release_delay)),
        }
        
        # Execute the action if it's not IDLE
        if action > 0 and action < len(Actions):
            action_func = action_map.get(action)
            if action_func:
                action_func()
            
        # Perform the game tick
        ticks = self.frame_skip
        if not self.debug:
            self.pyboy.tick(ticks, not self.headless, False)
        else:
            for tick in range(ticks):
                self.pyboy.tick(1, not self.headless, False)
        self._frames_played += ticks

        # Get game state
        self._calculate_fitness()

        # Determine if episode is done based on episode_mode
        if self.episode_mode == "life":
            # Episode ends when a single life is lost (immediate loss, not after saver)
            done = self._game_wrapper.lost_ball_during_saver 
        elif self.episode_mode == "ball":
            # Episode ends when a ball is completely lost (including after saver)
            done = self.ball_lost
        elif self.episode_mode == "game":
            # Episode ends when the game is over (all balls used)
            done = self._game_wrapper.game_over 
        else:
            # Default behavior from original code
            done = self._game_wrapper.lost_ball_during_saver or self._game_wrapper.game_over or self._game_wrapper.balls_left < 2 
        
        # Apply reward shaping by calling the reward function
        # but passing self for tracking state
        if self.reward_shaping:
            reward = self._apply_reward_shaping(self.reward_shaping)
        else:
            reward = self._fitness - self._previous_fitness
            
        observation = self._get_obs()
        
        # Check for stuck ball
        truncated = self._check_if_stuck(observation)
        
        info = self._get_info()
        
        # Add episode completion flag when episode ends
        if done or truncated:
            info['episode_complete'] = [True]
            info['episode_length'] = [float(self._frames_played)]
            
            # Add high score flag if appropriate (for tracking best episodes)
            if self._game_wrapper.score > self._high_score:
                self._high_score = self._game_wrapper.score
                info['high_score'] = [True]
                
            # Ensure all episode metrics are available in the final info dictionary
            total_ball_upgrades = (
                self._game_wrapper.great_ball_upgrades +
                self._game_wrapper.ultra_ball_upgrades +
                self._game_wrapper.master_ball_upgrades
            )
            info['total_ball_upgrades'] = [float(total_ball_upgrades)]
        
        # Add stuck status to info if applicable
        if truncated:
            info['truncated_reason'] = ['stuck_ball']
            
        return observation, reward, done, truncated, info
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment based on the reset_condition.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset reward tracking variables
        self._prev_caught = 0
        self._prev_evolutions = 0
        self._prev_stages_completed = 0
        self._prev_ball_upgrades = 0
        
        game_wrapper = self._game_wrapper
        
        # Mark the environment as initialized
        self._initialized = True
        game_wrapper.reset_tracking()
        
        # Implement reset condition logic
        if self.reset_condition == "life":
            # For "life" mode, we continue with the current ball after a life is lost
            if game_wrapper.game_over or self.life_lost:
                game_wrapper.reset_game()
            # Otherwise, just continue with the current ball
        elif self.reset_condition == "ball":
            # For "ball" mode, we start a new ball (or game) after a ball is lost
            if game_wrapper.game_over or self.ball_lost:
                game_wrapper.reset_game()
        elif self.reset_condition == "game":
            if game_wrapper.game_over:
                game_wrapper.reset_game()
        
        # Reset fitness tracking
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0
        
        # Reset stuck detection
        self.ball_position_history = []
        self.last_score = 0
        
        # Clear frame buffer
        buffer_key = f"{self.visual_mode}_buffer"
        if hasattr(self, buffer_key):
            delattr(self, buffer_key)
        
        # Track episode completion
        self.episodes_completed += 1
        self._episode_count += 1

        # Reset ball tracking to current state
        self._previous_balls_left = game_wrapper.balls_left
        self._previous_balls_lost_during_saver = game_wrapper.lost_ball_during_saver

        # Get observation and info
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
        
        # Add info about episode and reset modes
        info.update({
            'episode_mode': [self.episode_mode],
            'reset_condition': [self.reset_condition],
            'balls_left': [float(game_wrapper.balls_left)]
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
            # Use PyBoy's screen.ndarray but make a copy as recommended in the docs
            # "Remember to copy this object if you intend to store it"
            screen_img = np.array(self.pyboy.screen.ndarray[:,:,:3], copy=True)
            
            # Convert to grayscale using simple mean across channels
            screen_img = np.mean(screen_img, axis=2, keepdims=False).astype(np.uint8)
            
            if self.reduce_screen_resolution:
                screen_img = screen_img[::2, ::2]
            
            visual_obs = screen_img
        
        # Create observation dictionary with properly formatted numpy arrays
        observation = {
            "visual_representation": np.asarray(visual_obs, dtype=np.uint8),
        }
        
        # Level 0 - no info
        if self.info_level == 0:
            return observation.get('visual_representation')
        
        ball_x = float(game_wrapper.ball_x) 
        ball_y = float(game_wrapper.ball_y) 
        ball_x_velocity = float(game_wrapper.ball_x_velocity) 
        ball_y_velocity = float(game_wrapper.ball_y_velocity) 
        
        observation.update({
            "ball_x": np.array([ball_x], dtype=np.float32),
            "ball_y": np.array([ball_y], dtype=np.float32),
            "ball_x_velocity": np.array([ball_x_velocity], dtype=np.float32),
            "ball_y_velocity": np.array([ball_y_velocity], dtype=np.float32),
        })
        
        if self.info_level == 1:
            return observation
        
        # Level 2 - More detailed information
        if self.info_level >= 2:
            # Handle None values for enum lookups by providing default values
            current_stage_idx = STAGE_TO_INDEX.get(game_wrapper.current_stage, 0)
            ball_type_idx = BALL_TYPE_TO_INDEX.get(game_wrapper.ball_type, 0)
            
            # Handle possible None values for special mode
            special_mode = int(game_wrapper.special_mode)
            special_mode_active = int(game_wrapper.special_mode_active)
            
            observation.update({
                "current_stage": np.array([current_stage_idx], dtype=np.int32),
                "ball_type": np.array([ball_type_idx], dtype=np.int32),
                "special_mode": np.array([special_mode], dtype=np.int32),
                "special_mode_active": np.array([special_mode_active], dtype=np.int32),
            })
                
            # Convert boolean to int array for tensor compatibility
            saver_active = 1 if (game_wrapper.ball_saver_seconds_left > 0) else 0
            observation["saver_active"] = np.array([saver_active], dtype=np.int32)
            
        # Level 3 - Most detailed information
        if self.info_level >= 3:
            # Handle possible None value for pikachu_saver_charge
            pikachu_charge = int(game_wrapper.pikachu_saver_charge)
            observation["pikachu_saver_charge"] = np.array([pikachu_charge], dtype=np.int32)
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
            'prev_ball_upgrades': self._prev_ball_upgrades,
            'prev_ball_lost_during_saver': self._game_wrapper.lost_ball_during_saver,
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


