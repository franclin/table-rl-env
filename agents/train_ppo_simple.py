import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from table_env.environment import TableArrangementEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create the simplified environment
    env = TableArrangementEnv()
    env = Monitor(env, "logs/")
    
    eval_env = TableArrangementEnv()
    
    # Simple evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=500,
        deterministic=True,
        render=False
    )
    
    # PPO with standard hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.001,
        n_steps=512,
        batch_size=32,
        n_epochs=5,
        gamma=0.99,
        tensorboard_log=None
    )
    
    print("=" * 60)
    print("Training on SIMPLE environment (6 guests, 2 tables)")
    print("=" * 60)
    
    # Train
    model.learn(total_timesteps=20000, callback=eval_callback)
    
    model.save("models/ppo_simple")
    
    print("\n" + "=" * 60)
    print("Testing trained model...")
    print("=" * 60)
    
    # Test with visualization
    test_env = TableArrangementEnv(render_mode="human")
    obs, _ = test_env.reset()
    
    total_reward = 0
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        
        if terminated:
            print(f"\n🎉 SUCCESS! Found perfect arrangement in {step+1} steps!")
            print(f"Total reward: {total_reward}")
            break
        elif truncated:
            print(f"\n⏰ Reached max steps")
            print(f"Final reward: {total_reward}")
            break
    
    test_env.close()
    
    # Evaluate over 20 episodes
    print("\n" + "=" * 60)
    print("Final Evaluation (20 episodes)")
    print("=" * 60)
    
    eval_env = TableArrangementEnv()
    rewards = []
    successes = 0
    
    for episode in range(20):
        obs, _ = eval_env.reset()
        episode_reward = 0
        for _ in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            if terminated:
                successes += 1
                break
            if truncated:
                break
        rewards.append(episode_reward)
        print(f"Episode {episode+1}: reward = {episode_reward:.0f}")
    
    print(f"\n📊 Results:")
    print(f"   Average reward: {np.mean(rewards):.0f} +/- {np.std(rewards):.0f}")
    print(f"   Success rate (perfect solution): {successes/20 * 100}%")
    
    eval_env.close()

if __name__ == "__main__":
    import numpy as np
    main()