"""
Patch for clean_pufferl.py to add extended training functionality.
Apply these modifications to the original clean_pufferl.py script.
"""

import numpy as np
import torch
from collections import deque

from schedulers import PPOScheduler, KLEarlyStopping, ExperienceReplayBuffer

# Add this function to clean_pufferl.py
def patch_train_function(original_train_function):
    """
    Patch the original train function with extended training functionality.
    
    Args:
        original_train_function: The original train function
        
    Returns:
        Patched train function
    """
    def patched_train(data):
        """
        Patched training function with extended PPO features.
        """
        config, profile, experience = data.config, data.profile, data.experience
        
        # Apply parameter updates based on training progress if configured
        if hasattr(config, "scheduler") and config.scheduler is not None:
            params = config.scheduler.step(data.global_step)
            # Update parameters in config
            config.learning_rate = params['learning_rate']
            
            if hasattr(config, "ent_coef_decay") and config.ent_coef_decay:
                config.ent_coef = params['entropy_coef']
                
            if hasattr(config, "clip_coef_decay") and config.clip_coef_decay:
                config.clip_coef = params['clip_coef']
                
            # Log updated parameters
            if data.wandb is not None:
                data.wandb.log({
                    "hyperparams/learning_rate": params['learning_rate'],
                    "hyperparams/entropy_coef": params['entropy_coef'],
                    "hyperparams/clip_coef": params['clip_coef'],
                    "training/progress": params['progress'],
                })
        
        # Initialize losses
        data.losses = make_losses()
        losses = data.losses
        
        with profile.train_misc:
            idxs = experience.sort_training_data()
            dones_np = experience.dones_np[idxs]
            values_np = experience.values_np[idxs]
            rewards_np = experience.rewards_np[idxs]
            # TODO: bootstrap between segment bounds
            advantages_np = compute_gae(dones_np, values_np,
                rewards_np, config.gamma, config.gae_lambda)
            experience.flatten_batch(advantages_np)

        # Optimizing the policy and value network
        total_minibatches = experience.num_minibatches * config.update_epochs
        mean_pg_loss, mean_v_loss, mean_entropy_loss = 0, 0, 0
        mean_old_kl, mean_kl, mean_clipfrac = 0, 0, 0
        
        # Add early stopping based on KL divergence
        if hasattr(config, "target_kl") and config.target_kl is not None:
            # Default to simple threshold if no patience specified
            if not hasattr(config, "kl_early_stopping"):
                # Create early stopping with default parameters
                config.kl_early_stopping = KLEarlyStopping(
                    target_kl=config.target_kl
                )
        
        for epoch in range(config.update_epochs):
            lstm_state = None
            for mb in range(experience.num_minibatches):
                with profile.train_misc:
                    obs = experience.b_obs[mb]
                    obs = obs.to(config.device)
                    atn = experience.b_actions[mb]
                    log_probs = experience.b_logprobs[mb]
                    val = experience.b_values[mb]
                    adv = experience.b_advantages[mb]
                    ret = experience.b_returns[mb]

                with profile.train_forward:
                    if experience.lstm_h is not None:
                        _, newlogprob, entropy, newvalue, lstm_state = data.policy(
                            obs, state=lstm_state, action=atn)
                        lstm_state = (lstm_state[0].detach(), lstm_state[1].detach())
                    else:
                        _, newlogprob, entropy, newvalue = data.policy(
                            obs.reshape(-1, *data.vecenv.single_observation_space.shape),
                            action=atn,
                        )

                    if config.device == 'cuda':
                        torch.cuda.synchronize()

                with profile.train_misc:
                    logratio = newlogprob - log_probs.reshape(-1)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfrac = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

                    adv = adv.reshape(-1)
                    if config.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -adv * ratio
                    pg_loss2 = -adv * torch.clamp(
                        ratio, 1 - config.clip_coef, 1 + config.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if config.clip_vloss:
                        v_loss_unclipped = (newvalue - ret) ** 2
                        v_clipped = val + torch.clamp(
                            newvalue - val,
                            -config.vf_clip_coef,
                            config.vf_clip_coef,
                        )
                        v_loss_clipped = (v_clipped - ret) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - ret) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                with profile.learn:
                    data.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(data.policy.parameters(), config.max_grad_norm)
                    data.optimizer.step()
                    if config.device == 'cuda':
                        torch.cuda.synchronize()

                with profile.train_misc:
                    losses.policy_loss += pg_loss.item() / total_minibatches
                    losses.value_loss += v_loss.item() / total_minibatches
                    losses.entropy += entropy_loss.item() / total_minibatches
                    losses.old_approx_kl += old_approx_kl.item() / total_minibatches
                    losses.approx_kl += approx_kl.item() / total_minibatches
                    losses.clipfrac += clipfrac.item() / total_minibatches
                    
                # Check for recovery mode if KL divergence is too high
                if hasattr(config, "recovery_mode") and config.recovery_mode:
                    if approx_kl > config.target_kl * 2:
                        # Log recovery action
                        if data.wandb is not None:
                            data.wandb.log({
                                "recovery/activated": True,
                                "recovery/kl_divergence": approx_kl.item(),
                            })
                        # Temporarily reduce learning rate
                        for param_group in data.optimizer.param_groups:
                            param_group['lr'] *= 0.5
                            
                        # Add a log message
                        print(f"Warning: KL divergence {approx_kl.item():.4f} exceeds 2x target {config.target_kl:.4f}")
                        print("Implementing recovery measures...")
                    
            # Check for early stopping based on KL divergence
            if hasattr(config, "kl_early_stopping") and config.kl_early_stopping.check(approx_kl.item()):
                print(f"Early stopping at epoch {epoch+1}/{config.update_epochs} due to high KL divergence ({approx_kl.item():.4f})")
                if data.wandb is not None:
                    data.wandb.log({
                        "early_stopping/activated": True,
                        "early_stopping/epoch": epoch,
                        "early_stopping/kl_divergence": approx_kl.item(),
                    })
                break
                
            # Alternative simple early stopping based just on threshold
            elif config.target_kl is not None and approx_kl > config.target_kl:
                print(f"Early stopping at epoch {epoch+1}/{config.update_epochs} due to KL threshold")
                break

        with profile.train_misc:
            # Legacy LR annealing - only use if not using scheduler
            if config.anneal_lr and not hasattr(config, "scheduler"):
                frac = 1.0 - data.global_step / config.total_timesteps
                lrnow = frac * config.learning_rate
                data.optimizer.param_groups[0]["lr"] = lrnow

            # Compute explained variance
            y_pred = experience.values_np
            y_true = experience.returns_np
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            losses.explained_variance = explained_var
            data.epoch += 1

            done_training = data.global_step >= config.total_timesteps
            # TODO: better way to get episode return update without clogging dashboard
            # TODO: make this appear faster
            if done_training or profile.update(data):
                mean_and_log(data)
                print_dashboard(config.env, data.utilization, data.global_step, data.epoch,
                    profile, data.losses, data.stats, data.msg)
                data.stats = defaultdict(list)

            if data.epoch % config.checkpoint_interval == 0 or done_training:
                save_checkpoint(data)
                data.msg = f'Checkpoint saved at update {data.epoch}'
                
        # Store buffer data if using experience replay
        if hasattr(config, "buffer_stabilization") and config.buffer_stabilization:
            if not hasattr(data, "replay_buffer"):
                # Initialize replay buffer if it doesn't exist
                data.replay_buffer = ExperienceReplayBuffer(capacity=100000)
                
            # Add current batch to buffer with priority based on TD error
            td_error = np.abs(experience.returns_np - experience.values_np).mean()
            # Store relevant experience data
            experience_data = {
                "obs": experience.obs.detach().cpu(),
                "actions": experience.actions.detach().cpu(),
                "returns": torch.as_tensor(experience.returns_np),
                "advantages": torch.as_tensor(advantages_np),
                "td_error": td_error,
            }
            data.replay_buffer.add(experience_data, priority=td_error)
            
            # Log buffer stats
            if data.wandb is not None:
                data.wandb.log({
                    "buffer/size": len(data.replay_buffer),
                    "buffer/td_error": td_error,
                })
    
    return patched_train


# Modify the create function to initialize the scheduler
def patch_create_function(original_create_function):
    """
    Patch the original create function to initialize extended training features.
    
    Args:
        original_create_function: The original create function
        
    Returns:
        Patched create function
    """
    def patched_create(config, vecenv, policy, optimizer=None, wandb=None):
        """
        Patched create function that initializes extended training features.
        """
        # Call the original create function
        data = original_create_function(config, vecenv, policy, optimizer, wandb)
        
        # Initialize scheduler if needed
        if hasattr(config, "lr_decay") and config.lr_decay != "none":
            from schedulers import PPOScheduler
            
            data.scheduler = PPOScheduler(
                optimizer=data.optimizer,
                total_timesteps=config.total_timesteps,
                lr_decay=config.lr_decay,
                initial_lr=config.learning_rate,
                initial_ent_coef=config.ent_coef,
                initial_clip_coef=config.clip_coef,
                ent_coef_decay=config.ent_coef_decay if hasattr(config, "ent_coef_decay") else False,
                clip_coef_decay=config.clip_coef_decay if hasattr(config, "clip_coef_decay") else False,
            )
            # Store the scheduler in config for access during training
            config.scheduler = data.scheduler
            
        # Initialize KL early stopping
        if hasattr(config, "target_kl") and config.target_kl is not None:
            from schedulers import KLEarlyStopping
            
            config.kl_early_stopping = KLEarlyStopping(
                target_kl=config.target_kl,
                patience=3,  # Stop after 3 consecutive violations
                cooldown=5,  # Wait 5 updates before tracking again
            )
            
        return data
        
    return patched_create


def apply_patches():
    """
    Apply all patches to clean_pufferl.py functions.
    
    Usage:
    ```
    import clean_pufferl
    from clean_pufferl_patch import apply_patches
    
    # Apply patches
    clean_pufferl.train = apply_patches.patch_train_function(clean_pufferl.train)
    clean_pufferl.create = apply_patches.patch_create_function(clean_pufferl.create)
    ```
    """
    import clean_pufferl
    
    # Patch functions
    clean_pufferl.train = patch_train_function(clean_pufferl.train)
    clean_pufferl.create = patch_create_function(clean_pufferl.create)
    
    print("Successfully applied extended training patches to clean_pufferl.py")