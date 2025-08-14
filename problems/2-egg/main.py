import random


class EggGame:

    def __init__(self, n_eggs: int, n_floors: int, breaking_pnt: int):
        self.n_eggs_remaining = n_eggs
        self.drop_cnt = 0
        self.breaking_pnt = breaking_pnt
        self.n_floors = n_floors
        self.floors = []
        for i in range(n_floors):
            self.floors.append(i < breaking_pnt)

    def drop_egg(self, floor) -> int:
        """
        return 0: Egg did not break
        return 1: Egg broke
        return 2: Game is over
        return 3: Invalid floor
        """
        if floor < 0 or floor >= len(self.floors):
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
        return self.breaking_pnt == breaking_pnt




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_game = EggGame(2, 10, 6)
    print(test_game.get_state())
    print(test_game.drop_egg(0))
    print(test_game.drop_egg(5))
    print(test_game.drop_egg(11))
    print(test_game.drop_egg(6))
    print(test_game.drop_egg(8))
    print(test_game.drop_egg(8))
    print(test_game.get_state())
    print(test_game.guess_breaking_pnt(6))
    print(test_game.guess_breaking_pnt(5))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
