"""
Brute-force method demo for the Egg Dropping game.

This script imports `EggGame` and provides a `main()` that
demonstrates a simple linear search to identify the breaking point.
"""

from egg_game import EggGame


def brute_force_find_breaking_point(game: EggGame) -> int:
    """
    Linearly scan floors from 0 upward and drop an egg at each floor
    until an egg breaks. The first breaking floor is the breaking point.

    If no floor causes a break or other issues happen, return -1
    """
    n_floors = game.get_nbr_floors()
    for floor in range(n_floors):
        result = game.drop_egg(floor)
        if result == 1:
            # First break (1) indicates this floor is the breaking point.
            return floor
        elif result == 2 or result == 3:
            return -1
    # No break occurred => breaking point equals number of floors
    return -1


def main():
    # Example configuration; adjust as needed
    n_eggs = 2
    n_floors = 100
    breaking_pnt = 20

    game = EggGame(n_eggs=n_eggs, n_floors=n_floors, breaking_pnt=breaking_pnt)
    found_breaking_point = brute_force_find_breaking_point(game)

    # Single allowed guess: verify our found breaking point
    is_correct = game.guess_breaking_pnt(found_breaking_point)

    print(
        f"Found breaking point: {found_breaking_point} | "
        f"Correct: {is_correct} | State: {game.get_state()}"
    )


if __name__ == "__main__":
    main()
