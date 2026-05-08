def evaluate_model(model_path, env, n_episodes=50):
    model = PPO.load(model_path)
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards)