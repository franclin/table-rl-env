"""
Stable RL Environment - No reward hacks, proper episode management
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any, List

class TableArrangementEnv(gym.Env):
    """
    Stable environment with proper rewards and episode management.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Small problem size
        self.num_guests = 6
        self.num_tables = 2
        self.capacity = 3
        
        self.render_mode = render_mode
        
        # Simple constraints
        self.must_sit_pairs = [(0, 1), (2, 3)]
        self.avoid_pairs = [(0, 4)]
        
        # Action space: (guest_id, table_id)
        self.action_space = spaces.MultiDiscrete([self.num_guests, self.num_tables])
        
        # Observation space
        obs_dim = self.num_guests * self.num_tables
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_dim,), dtype=np.float32
        )
        
        # Track if episode is done
        self.assignment = None
        self.step_count = None
        self.episode_done = None
        self.last_action = None
        
        self.reset()
    
    def _get_obs(self) -> np.ndarray:
        return self.assignment.flatten().astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        table_counts = self.assignment.sum(axis=0)
        return {
            "table_counts": table_counts.astype(int).tolist(),
            "must_sit_satisfied": self._count_satisfied_must_sit(),
            "avoid_violations": self._count_violated_avoid_pairs(),
            "step_count": self.step_count
        }
    
    def _count_satisfied_must_sit(self) -> int:
        satisfied = 0
        for g1, g2 in self.must_sit_pairs:
            if np.argmax(self.assignment[g1]) == np.argmax(self.assignment[g2]):
                satisfied += 1
        return satisfied
    
    def _count_violated_avoid_pairs(self) -> int:
        violations = 0
        for g1, g2 in self.avoid_pairs:
            if np.argmax(self.assignment[g1]) == np.argmax(self.assignment[g2]):
                violations += 1
        return violations
    
    def _is_capacity_violated(self) -> bool:
        """Check if any table exceeds capacity."""
        table_counts = self.assignment.sum(axis=0)
        return np.any(table_counts > self.capacity)
    
    def _is_perfect(self) -> bool:
        """Check if perfect solution achieved."""
        # All must-sit pairs satisfied
        if self._count_satisfied_must_sit() != len(self.must_sit_pairs):
            return False
        # No avoid violations
        if self._count_violated_avoid_pairs() > 0:
            return False
        # No capacity violations
        if self._is_capacity_violated():
            return False
        return True
    
    def _compute_reward(self) -> float:
        """
        Sparse reward - only at episode end.
        This prevents reward hacks and encourages exploration.
        """
        # Only give reward at end of episode
        if not (self.episode_done or self.step_count >= 50):
            return 0.0
        
        # Calculate final score
        reward = 0.0
        
        # Must-sit pairs: +100 each
        satisfied = self._count_satisfied_must_sit()
        reward += satisfied * 100.0
        
        # Avoid violations: -100 each
        violations = self._count_violated_avoid_pairs()
        reward += violations * -100.0
        
        # Capacity violations: -200
        if self._is_capacity_violated():
            reward -= 200.0
        
        # Perfect solution bonus: +500
        if self._is_perfect():
            reward += 500.0
        
        return reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        
        # Start with all guests at table 0 (structured start)
        self.assignment = np.zeros((self.num_guests, self.num_tables), dtype=np.float32)
        self.assignment[:, 0] = 1.0
        
        self.step_count = 0
        self.episode_done = False
        self.last_action = None
        
        return self._get_obs(), self._get_info()
    
    def step(self, action: np.ndarray):
        """Take a step. Reward only given when episode ends."""
        
        if self.episode_done:
            # Episode already finished, reset needed
            return self._get_obs(), 0.0, True, False, self._get_info()
        
        guest_id, target_table = int(action[0]), int(action[1])
        
        # Clip to valid ranges
        guest_id = min(max(guest_id, 0), self.num_guests - 1)
        target_table = min(max(target_table, 0), self.num_tables - 1)
        
        # Move guest
        self.assignment[guest_id, :] = 0.0
        self.assignment[guest_id, target_table] = 1.0
        
        self.step_count += 1
        self.last_action = (guest_id, target_table)
        
        # Check if episode should end
        terminated = self._is_perfect()
        truncated = self.step_count >= 50
        
        self.episode_done = terminated or truncated
        
        # Compute reward (only at end of episode)
        reward = self._compute_reward() if self.episode_done else 0.0
        
        if self.render_mode == "human" and (terminated or truncated):
            self.render()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Print current state."""
        print(f"\n{'='*40}")
        print(f"Episode finished at step {self.step_count}")
        print(f"Final reward: {self._compute_reward():.0f}")
        print(f"{'='*40}")
        
        for t in range(self.num_tables):
            guests = np.where(self.assignment[:, t] == 1.0)[0]
            capacity_status = f"({len(guests)}/{self.capacity})"
            if len(guests) > self.capacity:
                capacity_status += " ⚠️ OVER CAPACITY"
            print(f"Table {t}: {capacity_status} Guests: {guests.tolist()}")
        
        satisfied = self._count_satisfied_must_sit()
        total_must = len(self.must_sit_pairs)
        violations = self._count_violated_avoid_pairs()
        
        print(f"\nMust-sit pairs satisfied: {satisfied}/{total_must}")
        print(f"Avoid pair violations: {violations}")
        
        if self._is_perfect():
            print("\n🎉 PERFECT SOLUTION! 🎉")
    
    def close(self):
        pass