import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from table_env.environment import TableArrangementEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import numpy as np

def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environment with OPTIMIZED settings
    env = TableArrangementEnv(
        num_guests=12,          # Smaller problem = faster learning
        num_tables=4,
        capacity_per_table=3,
        render_mode=None
    )
    
    # Store environment parameters before wrapping
    num_guests = env.num_guests
    num_tables = env.num_tables
    
    # Wrap with Monitor for logging
    env = Monitor(env, "logs/")
    
    # Evaluation environment
    eval_env = TableArrangementEnv(
        num_guests=12,
        num_tables=4,
        capacity_per_table=3,
        render_mode=None
    )
    
    # Stop when reward > 3
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=3.0, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=1000,
        deterministic=True,
        render=False,
        callback_on_new_best=callback_on_best
    )
    
    # OPTIMIZED hyperparameters for this environment
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0005,
        n_steps=1024,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=None  # Disable TensorBoard
    )
    
    print("Starting training with OPTIMIZED settings...")
    print(f"Environment: {num_guests} guests, {num_tables} tables")
    print("=" * 60)
    
    # Train
    model.learn(
        total_timesteps=30000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save model
    model.save("models/ppo_table_arranger_final")
    
    print("=" * 60)
    print("Training complete!")
    
    # Test the trained model
    print("\nTesting trained model...")
    test_env = TableArrangementEnv(
        num_guests=12,
        num_tables=4,
        capacity_per_table=3,
        render_mode="human"
    )
    
    obs, _ = test_env.reset()
    total_reward = 0
    step_count = 0
    
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        step_count += 1
        
        if terminated:
            print(f"\n✅ Found good solution in {step_count} steps!")
            print(f"Total reward: {total_reward:.2f}")
            break
        elif truncated:
            print(f"\n⏰ Episode truncated at {step_count} steps")
            print(f"Final reward: {total_reward:.2f}")
            break
    
    test_env.close()
    
    # Print final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    # Recreate fresh eval environment
    final_eval_env = TableArrangementEnv(
        num_guests=12,
        num_tables=4,
        capacity_per_table=3,
        render_mode=None
    )
    
    eval_rewards = []
    for episode in range(10):
        obs, _ = final_eval_env.reset()
        episode_reward = 0
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = final_eval_env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        eval_rewards.append(episode_reward)
        print(f"Episode {episode+1}: {episode_reward:.2f}")
    
    print(f"\nAverage reward over 10 episodes: {np.mean(eval_rewards):.2f} +/- {np.std(eval_rewards):.2f}")
    final_eval_env.close()

if __name__ == "__main__":
    main()