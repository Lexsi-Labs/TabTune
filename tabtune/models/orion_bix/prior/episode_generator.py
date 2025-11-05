"""
Episode generator for meta-learning with optimized performance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import random


class EpisodeGenerator:
    """Generates episodes for meta-learning with optimized performance."""

    def __init__(
        self,
        support_size: int = 32,
        query_size: int = 32,
        k_neighbors: int = 64,
        similarity_metric: str = "cosine",
        feature_normalization: bool = True,
        num_episodes_per_dataset: int = 20,
        device: str = "cpu",
        diversity_factor: float = 0.3,
        min_dataset_size: int = 128,
        episode_method: str = "random",
    ):
        self.support_size = support_size
        self.query_size = query_size
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.feature_normalization = feature_normalization
        self.num_episodes_per_dataset = num_episodes_per_dataset
        self.device = device
        self.diversity_factor = diversity_factor
        self.min_dataset_size = min_dataset_size
        self.episode_method = episode_method
        # FIXED: Create a single generator instance
        self.rng = torch.Generator()
        
        if episode_method not in ["random", "knn"]:
            raise ValueError(f"episode_method must be 'random' or 'knn', got {episode_method}")
    
    def compute_similarity(self, query_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
        """Compute similarity between query and support features - OPTIMIZED."""
        if self.feature_normalization:
            query_features = F.normalize(query_features, p=2, dim=1)
            support_features = F.normalize(support_features, p=2, dim=1)
        
        if self.similarity_metric == "cosine":
            return torch.mm(query_features, support_features.t())
        elif self.similarity_metric == "euclidean":
            return -torch.cdist(query_features, support_features, p=2)
        elif self.similarity_metric == "manhattan":
            return -torch.cdist(query_features, support_features, p=1)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")
    
    def select_diverse_knn_support(self, query_features: torch.Tensor, support_features: torch.Tensor, 
                                 support_labels: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select diverse k-NN support samples - OPTIMIZED."""
        similarities = self.compute_similarity(query_features, support_features)
        
        # OPTIMIZED: Use vectorized operations instead of loops
        if self.diversity_factor > 0:
            # Compute pairwise similarities once
            support_similarities = torch.mm(support_features, support_features.t())
            
            # Vectorized diversity penalty
            diversity_penalty = torch.zeros_like(similarities)
            if similarities.size(0) == 1:  # Single query case
                # Use upper triangular matrix to avoid self-similarity
                upper_tri = torch.triu(support_similarities, diagonal=1)
                max_similarities = torch.max(upper_tri, dim=1)[0]
                diversity_penalty[0] = self.diversity_factor * max_similarities
            
            similarities = similarities - diversity_penalty
        
        _, top_k_indices = torch.topk(similarities, k=min(k, support_features.size(0)), dim=1)
        
        # Gather selected support samples
        selected_support_features = support_features[top_k_indices.flatten()].view(query_features.size(0), -1, support_features.size(1))
        selected_support_labels = support_labels[top_k_indices.flatten()].view(query_features.size(0), -1)
        
        return selected_support_features, selected_support_labels
    
    def generate_episodes_from_batch(self, X: torch.Tensor, y: torch.Tensor, d: torch.Tensor, 
                                   seq_lens: torch.Tensor, train_sizes: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate episodes using the specified method - OPTIMIZED."""
        if self.episode_method == "random":
            return self._generate_random_episodes(X, y, d, seq_lens, train_sizes)
        elif self.episode_method == "knn":
            return self._generate_knn_episodes(X, y, d, seq_lens, train_sizes)
        else:
            raise ValueError(f"Unknown episode method: {self.episode_method}")
    
    def _generate_random_episodes(self, X: torch.Tensor, y: torch.Tensor, d: torch.Tensor, 
                                          seq_lens: torch.Tensor, train_sizes: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate episodes using optimized random splits."""
        episodes = []
        
        # OPTIMIZATION: Pre-generate all random seeds
        total_episodes = X.size(0) * self.num_episodes_per_dataset
        #rng = torch.Generator()
        
        for dataset_idx in range(X.size(0)):
            dataset_X = X[dataset_idx, :seq_lens[dataset_idx]]
            dataset_y = y[dataset_idx, :seq_lens[dataset_idx]]
            dataset_seq_len = seq_lens[dataset_idx].item()
            
            if dataset_seq_len < self.min_dataset_size:
                continue
            
            min_required = self.support_size + self.query_size
            if dataset_seq_len < min_required:
                continue
            
            # OPTIMIZATION: Generate all episodes for this dataset at once
            for episode_idx in range(self.num_episodes_per_dataset):
                try:
                    # OPTIMIZATION: Use deterministic but fast random generation
                    self.rng.manual_seed(episode_idx + dataset_idx * 1000)
                    all_indices = torch.randperm(dataset_seq_len, generator=self.rng)
                    
                    support_indices = all_indices[:self.support_size]
                    query_indices = all_indices[self.support_size:self.support_size + self.query_size]
                    
                    support_X = dataset_X[support_indices]
                    support_y = dataset_y[support_indices]
                    query_X = dataset_X[query_indices]
                    query_y = dataset_y[query_indices]
                    
                    episode = {
                        'support_X': support_X,
                        'support_y': support_y,
                        'query_X': query_X,
                        'query_y': query_y,
                        'd': d[dataset_idx],
                        'seq_len': dataset_seq_len,
                        'support_size': self.support_size,
                        'episode_idx': episode_idx,
                        'dataset_idx': dataset_idx,
                        'method': 'random'
                    }
                    episodes.append(episode)
                    
                except Exception as e:
                    continue
        
        return episodes
    
    def _generate_knn_episodes(self, X: torch.Tensor, y: torch.Tensor, d: torch.Tensor, 
                                        seq_lens: torch.Tensor, train_sizes: torch.Tensor) -> List[Dict[str, Any]]:
        """Generate episodes using optimized k-NN support selection."""
        episodes = []
        
        for dataset_idx in range(X.size(0)):
            dataset_X = X[dataset_idx, :seq_lens[dataset_idx]]
            dataset_y = y[dataset_idx, :seq_lens[dataset_idx]]
            dataset_seq_len = seq_lens[dataset_idx].item()
            
            if dataset_seq_len < self.min_dataset_size:
                continue
            
            # OPTIMIZATION: Generate all episodes for this dataset at once
            for episode_idx in range(self.num_episodes_per_dataset):
                try:
                    # Use deterministic random generation
                    #rng = torch.Generator()
                    self.rng.manual_seed(episode_idx + dataset_idx * 1000)
                    all_indices = torch.randperm(dataset_seq_len, generator=self.rng)
                    
                    query_indices = all_indices[:self.query_size]
                    remaining_indices = all_indices[self.query_size:]
                    
                    if len(remaining_indices) < self.support_size:
                        continue
                    
                    query_X = dataset_X[query_indices]
                    query_y = dataset_y[query_indices]
                    support_candidates_X = dataset_X[remaining_indices]
                    support_candidates_y = dataset_y[remaining_indices]
                    
                    # OPTIMIZATION: Use mean of query features for k-NN
                    query_mean = query_X.mean(dim=0, keepdim=True)
                    
                    selected_features, selected_labels = self.select_diverse_knn_support(
                        query_mean, support_candidates_X, support_candidates_y, self.support_size
                    )
                    
                    support_X = selected_features[0, :self.support_size, :]
                    support_y = selected_labels[0, :self.support_size]
                    
                    episode = {
                        'support_X': support_X,
                        'support_y': support_y,
                        'query_X': query_X,
                        'query_y': query_y,
                        'd': d[dataset_idx],
                        'seq_len': dataset_seq_len,
                        'support_size': self.support_size,
                        'episode_idx': episode_idx,
                        'dataset_idx': dataset_idx,
                        'method': 'knn'
                    }
                    episodes.append(episode)
                    
                except Exception as e:
                    continue
        
        return episodes