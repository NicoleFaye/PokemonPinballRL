"""
Reward shaping functions for Pokemon Pinball environment.
"""

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