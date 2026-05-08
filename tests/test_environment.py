def test_env_creation():
    from table_env.environment import TableArrangementEnv
    env = TableArrangementEnv(num_guests=10, num_tables=3)
    assert env.reset() is not None