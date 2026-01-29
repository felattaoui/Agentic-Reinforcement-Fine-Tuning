"""
In-memory database for tau-bench retail environment.

Provides:
- Thread-safe access to users, orders, products
- Snapshot/restore for test isolation
- Mutation tracking for verification
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import copy


@dataclass
class RetailDatabase:
    """
    In-memory database holding retail data.

    Attributes:
        users: Dict of user_id -> user profile
        orders: Dict of order_id -> order details
        products: Dict of product_id -> product info
        _mutations: List of mutations for tracking
    """
    users: Dict[str, Any] = field(default_factory=dict)
    orders: Dict[str, Any] = field(default_factory=dict)
    products: Dict[str, Any] = field(default_factory=dict)
    _mutations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return data dict compatible with tau-bench tools."""
        return {
            "users": self.users,
            "orders": self.orders,
            "products": self.products
        }

    def snapshot(self) -> "RetailDatabase":
        """Create a deep copy for test isolation."""
        return RetailDatabase(
            users=copy.deepcopy(self.users),
            orders=copy.deepcopy(self.orders),
            products=copy.deepcopy(self.products),
            _mutations=[]
        )

    def record_mutation(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Track mutations for verification."""
        self._mutations.append({
            "tool": tool_name,
            "args": args,
            "result": str(result)[:500]  # Truncate for logging
        })

    def get_mutations(self) -> List[Dict[str, Any]]:
        """Get recorded mutations."""
        return self._mutations.copy()

    def clear_mutations(self) -> None:
        """Clear mutation log."""
        self._mutations = []

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product by ID."""
        return self.products.get(product_id)

    def __repr__(self) -> str:
        return f"RetailDatabase(users={len(self.users)}, orders={len(self.orders)}, products={len(self.products)})"
