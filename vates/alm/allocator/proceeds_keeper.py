class ProceedsKeeper:

    __slots__= ('_balance',)

    def __init__(self, init_balance: float = 0.0):
        self._balance: float = init_balance

    @property
    def balance(self) -> float:
        return self._balance

    def __iadd__(self, amount: float):
        self._balance += amount
        return self

    def __isub__(self, amount: float):
        self._balance -= amount
        return self

    def take_out(self) -> float:
        amount, self._balance = self._balance, 0.0
        return amount
