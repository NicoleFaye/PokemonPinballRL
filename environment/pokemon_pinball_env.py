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
    "visual_mode": "game_area", # alternative would be screen
    "frame_stack_extra_observation": False,
    "reduce_screen_resolution": True,
}


class PokemonPinballEnv(gym.Env):
    """Pokemon Pinball environment for reinforcement learning."""
    
    metadata = {"render_modes": ["human"]}
    
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
        self.pyboy = PyBoy(rom_path, debug=config['debug'], headless=config['headless'])
        if self.pyboy is None:
            raise ValueError("PyBoy instance is required")
        assert self.pyboy.cartridge_title == "POKEPINBALLVPH", "Invalid ROM: Pokémon Pinball required"
        
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0  # Track frames played in current episode
        
        self.debug = config['debug']
        self.headless = config['headless']

        self.frame_skip = config['frame_skip']

        self.frame_stack = config['frame_stack']
        self.frame_stack_extra_observation = config['frame_stack_extra_observation']

        if config['visual_mode'] == "game_area":
            self.output_shape = (16, 20, self.frame_stack)
        elif config['visual_mode'] == "screen":
            #TODO implement screen mode
            if config['reduce_screen_resolution']:
                self.output_shape = (144/2, 160/2, self.frame_stack)
            else:
                self.output_shape = (144, 160, self.frame_stack)

        
        # Configure speed based on mode
        if self.debug:
            # Normal speed for debugging
            self.pyboy.set_emulation_speed(1.0)
        else:
            # Maximum speed (0 = no limit)
            self.pyboy.set_emulation_speed(0)
            
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = self._create_observation_space(config['info_level'])
        
        self.reward_shaping = config['reward_shaping']
        self.info_level = config['info_level']
        
        self._game_wrapper = self.pyboy.game_wrapper
        
        # Initialize game
        self._game_wrapper.start_game()

    def _create_observation_space(self, info_level):
        """Create an observation space based on the info level."""
        # Base space is always the game area
        observations_dict = {}

        observations_dict.update( {
            'game_area': spaces.Box(low=0, high=255, shape=self.output_shape ,dtype=np.uint8)
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
        done = self._game_wrapper.game_over
        
        # Apply reward shaping
        if self.reward_shaping:
            reward = self.reward_shaping(self._fitness, self._previous_fitness, self._game_wrapper, self._frames_played)
        else:
            reward = self._fitness - self._previous_fitness
            
        # Get observation
        observation = self._get_obs()
        
        # Get info with appropriate level of detail
        info = self._get_info()
        
        # Check if the game is truncated (cut short for some reason)
        # Always false for now, could be parameterized later
        truncated = False
        
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
        
        # Reset reward tracking variables
        RewardShaping._prev_caught = 0
        RewardShaping._prev_evolutions = 0
        RewardShaping._prev_stages_completed = 0
        RewardShaping._prev_ball_upgrades = 0
        
        game_wrapper = self._game_wrapper
        game_wrapper.reset_game()
        # this method currently is not in the official pyboy API, but there is a pull request to add it
        game_wrapper.reset_tracking()
        
        # Reset fitness tracking
        self._fitness = 0
        self._previous_fitness = 0
        self._frames_played = 0  # Reset frame counter
        
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
        
    def _get_info(self):
        """Get additional information from the environment."""
        return {}
        
    def _get_obs(self):
        """
        Get the current observation from the environment.
        """
        game_wrapper = self._game_wrapper
        
        observation = {
            "game_area": game_wrapper.game_area(),
        }
        # Level 0 - no info
        if self.info_level == 0:
            return observation
        
        # Level 1 - position and velocity information
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


class RewardShaping:
    """
    Collection of reward shaping functions for Pokemon Pinball.
    These can be passed to the environment to modify the reward structure.
    """
    
    # Class-level tracking variables for reward shaping
    _prev_caught = 0
    _prev_evolutions = 0
    _prev_stages_completed = 0
    _prev_ball_upgrades = 0
    
    @staticmethod
    def basic(current_fitness, previous_fitness, game_wrapper, frames_played=0):
        """Basic reward shaping based on score difference."""
        return current_fitness - previous_fitness
        
    @classmethod
    def catch_focused(cls, current_fitness, previous_fitness, game_wrapper, frames_played=0):
        """Reward focused on catching Pokemon."""
        score_reward = (current_fitness - previous_fitness) * 0.5
        
        # Big reward for catching Pokemon
        catch_reward = 0
        if game_wrapper.pokemon_caught_in_session > cls._prev_caught:
            catch_reward = 1000
            cls._prev_caught = game_wrapper.pokemon_caught_in_session
            
        return score_reward + catch_reward
        
    @classmethod
    def comprehensive(cls, current_fitness, previous_fitness, game_wrapper, frames_played=0):
        """Comprehensive reward that promotes long survival and steady progress."""
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
        if game_wrapper.pokemon_caught_in_session > cls._prev_caught:
            pokemon_reward = 500
            additional_reward += pokemon_reward
            reward_sources["pokemon_catch"] = pokemon_reward
            cls._prev_caught = game_wrapper.pokemon_caught_in_session

        # Evolution rewards
        if game_wrapper.evolution_success_count > cls._prev_evolutions:
            evolution_reward = 1000
            additional_reward += evolution_reward
            reward_sources["evolution"] = evolution_reward
            cls._prev_evolutions = game_wrapper.evolution_success_count

        # Stage completion
        total_stages_completed = (
            game_wrapper.diglett_stages_completed +
            game_wrapper.gengar_stages_completed +
            game_wrapper.meowth_stages_completed +
            game_wrapper.seel_stages_completed +
            game_wrapper.mewtwo_stages_completed
        )
        if total_stages_completed > cls._prev_stages_completed:
            stage_reward = 1500
            additional_reward += stage_reward
            reward_sources["stage_completion"] = stage_reward
            cls._prev_stages_completed = total_stages_completed

        # Ball upgrades
        ball_upgrades = (
            game_wrapper.great_ball_upgrades +
            game_wrapper.ultra_ball_upgrades +
            game_wrapper.master_ball_upgrades
        )
        if ball_upgrades > cls._prev_ball_upgrades:
            upgrade_reward = 200
            additional_reward += upgrade_reward
            reward_sources["ball_upgrade"] = upgrade_reward
            cls._prev_ball_upgrades = ball_upgrades

        # Combine all rewards
        total_reward = score_reward + additional_reward + ball_alive_reward + time_bonus

        return total_reward