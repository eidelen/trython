class EggGame:

    def __init__(self, n_eggs: int, n_floors: int, breaking_pnt: int):
        self.n_eggs_remaining = n_eggs
        self.drop_cnt = 0
        self.breaking_pnt = breaking_pnt
        self.n_floors = n_floors
        self.floors = []
        self._has_guessed = False
        for i in range(n_floors):
            self.floors.append(i < breaking_pnt)

    def get_nbr_floors(self) -> int:
        return self.n_floors

    def drop_egg(self, floor) -> int:
        """
        return 0: Egg did not break
        return 1: Egg broke
        return 2: Game is over
        return 3: Invalid floor
        """
        if floor < 0 or floor >= self.get_nbr_floors():
            return 3
        if self.n_eggs_remaining == 0:
            return 2

        self.drop_cnt += 1

        if not self.floors[floor]:
            self.n_eggs_remaining -= 1
            if self.n_eggs_remaining == 0:
                return 2
            return 1
        return 0

    def get_state(self):
        return self.n_floors, self.breaking_pnt, self.n_eggs_remaining, self.drop_cnt

    def guess_breaking_pnt(self, breaking_pnt: int):
        if self._has_guessed:
            raise RuntimeError("Breaking point has already been guessed once")
        self._has_guessed = True
        return self.breaking_pnt == breaking_pnt
