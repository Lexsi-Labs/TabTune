"""
Meta-learning dataset that generates episodes from standard OrionBix batches.
"""

import torch
from torch.utils.data import IterableDataset
from typing import List, Dict, Any, Tuple
from .episode_generator import EpisodeGenerator
from .dataset import PriorDataset
from .genload import LoadPriorDataset


class MetaLearningDataset(IterableDataset):
    """Meta-learning dataset that generates episodes from standard OrionBix batches."""
    
    def __init__(
        self,
        base_prior: PriorDataset | LoadPriorDataset,
        episode_generator: EpisodeGenerator,
        episodes_per_yield: int = 1280,  # Yield episodes in smaller chunks
    ):
        self.base_prior = base_prior
        self.episode_generator = episode_generator
        self.episodes_per_yield = episodes_per_yield
        self.current_batch = None
        self.current_episodes = []
        self.episode_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> List[Dict[str, Any]]:
        """Generate next batch of episodes - FIXED."""
        # If we have episodes left, return them in chunks
        if self.episode_idx < len(self.current_episodes):
            end_idx = min(self.episode_idx + self.episodes_per_yield, len(self.current_episodes))
            episodes = self.current_episodes[self.episode_idx:end_idx]
            self.episode_idx = end_idx
            return episodes
        
        # Get next standard OrionBix batch
        X, y, d, seq_lens, train_sizes = next(self.base_prior)
        
        # Generate ALL episodes from this batch at once
        self.current_episodes = self.episode_generator.generate_episodes_from_batch(
            X, y, d, seq_lens, train_sizes
        )
        
        if not self.current_episodes:
            return self.__next__()
        
        # FIXED: Reset episode_idx to 0 and return first chunk
        self.episode_idx = 0
        end_idx = min(self.episodes_per_yield, len(self.current_episodes))
        self.episode_idx = end_idx
        return self.current_episodes[0:end_idx]
        