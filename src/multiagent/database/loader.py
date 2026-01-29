"""
Load tau-bench retail data from JSON files.
"""

import json
from pathlib import Path
from typing import Optional

from .store import RetailDatabase


# Path to tau-bench data in project
PROJECT_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "tau_bench"


def load_database(data_path: Optional[Path] = None) -> RetailDatabase:
    """
    Load retail database from JSON files.

    Args:
        data_path: Path to directory containing users.json, orders.json, products.json
                   Defaults to data/tau_bench/ in project

    Returns:
        RetailDatabase instance with loaded data

    Raises:
        FileNotFoundError: If data files don't exist
    """
    if data_path is None:
        data_path = PROJECT_DATA_PATH

    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_path}\n"
            f"Run copy_tau_bench_data() to copy data from tau-bench package."
        )

    users_file = data_path / "users.json"
    orders_file = data_path / "orders.json"
    products_file = data_path / "products.json"

    for f in [users_file, orders_file, products_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing data file: {f}")

    users = json.loads(users_file.read_text(encoding="utf-8"))
    orders = json.loads(orders_file.read_text(encoding="utf-8"))
    products = json.loads(products_file.read_text(encoding="utf-8"))

    return RetailDatabase(users=users, orders=orders, products=products)


def copy_tau_bench_data(dest_path: Optional[Path] = None) -> None:
    """
    Copy tau-bench data from installed package to project directory.

    This allows the project to work independently of tau-bench installation location.

    Args:
        dest_path: Destination directory (defaults to data/tau_bench/)
    """
    if dest_path is None:
        dest_path = PROJECT_DATA_PATH

    dest_path = Path(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Try to find tau-bench data in common locations
    possible_sources = [
        # Conda environment
        Path(__file__).parent.parent.parent.parent / ".conda" / "Lib" / "site-packages" / "tau_bench" / "envs" / "retail" / "data",
        # Standard pip install
        Path(__file__).parent.parent.parent.parent / "venv" / "Lib" / "site-packages" / "tau_bench" / "envs" / "retail" / "data",
    ]

    # Try importing tau_bench to find its location
    try:
        import tau_bench
        tau_bench_path = Path(tau_bench.__file__).parent / "envs" / "retail" / "data"
        possible_sources.insert(0, tau_bench_path)
    except ImportError:
        pass

    source_path = None
    for p in possible_sources:
        if p.exists():
            source_path = p
            break

    if source_path is None:
        raise FileNotFoundError(
            "Could not find tau-bench data. Make sure tau-bench is installed:\n"
            "pip install tau-bench"
        )

    for filename in ["users.json", "orders.json", "products.json"]:
        src = source_path / filename
        dst = dest_path / filename
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Copied {filename} to {dest_path}")
        else:
            print(f"Warning: {src} not found")

    print(f"\nData copied to {dest_path}")
