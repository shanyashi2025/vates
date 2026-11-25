class UnallocatedCash:
    """Mutable balance for cash that has not been allocated to an asset."""

    __slots__= ('_balance',)

    def __init__(self, init_balance: float = 0.0):
        """Initialize the unallocated-cash balance."""
        self._balance: float = init_balance

    @property
    def balance(self) -> float:
        """Current unallocated-cash balance."""
        return self._balance

    def accrue_interest(self, rate: float) -> None:
        """Compound the current balance by a periodic interest rate."""
        self._balance *= (1 + rate)

    def __iadd__(self, amount: float):
        self._balance += amount
        return self

    def __isub__(self, amount: float):
        self._balance -= amount
        return self
