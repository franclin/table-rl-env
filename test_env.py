from table_env import TableArrangementEnv

def test_basic_behavior():
    env = TableArrangementEnv(num_guests=6, num_tables=2, capacity_per_table=3)
    
    obs, info = env.reset()
    assert obs.shape == (6 * 2,)
    assert info['capacity_violations'] >= 0
    
    # Test valid action
    obs, reward, terminated, truncated, info = env.step([0, 1])
    assert isinstance(reward, float)
    
    print("All tests passed!")

def test_reward_shapes():
    env = TableArrangementEnv(num_guests=4, num_tables=2, capacity_per_table=2)
    env.reset()
    
    # Move all guests to table 0 (should cause capacity violation)
    for guest in range(4):
        env.step([guest, 0])
    
    obs, reward, terminated, truncated, info = env.step([0, 0])
    assert reward < 0  # Should be negative due to capacity
    
    print("Reward test passed!")

if __name__ == "__main__":
    test_basic_behavior()
    test_reward_shapes()
    print("Environment ready for training!")