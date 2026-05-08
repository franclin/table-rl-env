import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from table_env.environment import TableArrangementEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np

class SuccessRateCallback(BaseCallback):
    """Track success rate during training."""
    
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.successes = []
        
    def _on_step(self) -> bool:
        if self.n_calls % 2000 == 0:
            # Evaluate success rate
            successes = 0
            for _ in range(20):
                obs, _ = self.eval_env.reset()
                terminated, truncated = False, False
                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                # Check if perfect solution
                if self.eval_env._is_perfect():
                    successes += 1
            success_rate = successes / 20 * 100
            self.successes.append(success_rate)
            print(f"\n📊 Step {self.n_calls}: Success rate = {success_rate}%")
        return True

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environments
    env = TableArrangementEnv()
    env = Monitor(env, "logs/")
    
    eval_env = TableArrangementEnv()
    
    # PPO with good hyperparameters for sparse rewards
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,  # Encourage exploration
        tensorboard_log=None
    )
    
    # Custom callback to track success
    success_callback = SuccessRateCallback(eval_env)
    
    print("=" * 60)
    print("Training with SPARSE REWARDS (reward only at episode end)")
    print("Goal: Perfect solution = 1000 points")
    print("=" * 60)
    
    # Train for 50k steps
    model.learn(total_timesteps=50000, callback=success_callback)
    
    model.save("models/ppo_stable")
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (100 episodes)")
    print("=" * 60)
    
    # Thorough evaluation
    test_env = TableArrangementEnv(render_mode="human")
    rewards = []
    perfect_count = 0
    
    for episode in range(100):
        obs, _ = test_env.reset()
        episode_reward = 0
        step_count = 0
        
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            step_count += 1
            
            if terminated:
                perfect_count += 1
                print(f"\n✨ Episode {episode+1}: PERFECT in {step_count} steps! (Reward: {episode_reward})")
                break
            if truncated:
                print(f"Episode {episode+1}: Failed - {step_count} steps, reward {episode_reward}")
                break
        
        rewards.append(episode_reward)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Perfect solutions: {perfect_count}/100 ({perfect_count}%)")
    print(f"Average reward: {np.mean(rewards):.0f} +/- {np.std(rewards):.0f}")
    
    # Show one successful example
    if perfect_count > 0:
        print("\n" + "=" * 60)
        print("EXAMPLE SUCCESSFUL ARRANGEMENT")
        print("=" * 60)
        
        # Find and show a successful episode
        obs, _ = test_env.reset()
        test_env.render_mode = "human"
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            if terminated:
                test_env.render()
                break
    
    test_env.close()

if __name__ == "__main__":
    main()