"""
Data utilities for loading and saving JSONL files.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(samples: List[Dict[str, Any]], path: Path) -> None:
    """
    Save a list of dictionaries to a JSONL file.
    
    Args:
        samples: List of dictionaries to save
        path: Path to the output file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def load_train_val_test(data_dir: Path) -> tuple:
    """
    Load train, validation, and test splits from a directory.
    
    Args:
        data_dir: Directory containing train.jsonl, val.jsonl, test.jsonl
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    data_dir = Path(data_dir)
    
    train_samples = load_jsonl(data_dir / "train.jsonl")
    val_samples = load_jsonl(data_dir / "val.jsonl")
    test_samples = load_jsonl(data_dir / "test.jsonl")
    
    return train_samples, val_samples, test_samples


def print_data_stats(train: list, val: list, test: list) -> None:
    """Print statistics about the data splits."""
    print(f"📦 Data loaded:")
    print(f"   Train: {len(train)} samples")
    print(f"   Val:   {len(val)} samples")
    print(f"   Test:  {len(test)} samples")
    print(f"   Total: {len(train) + len(val) + len(test)} samples")
