"""
Brute-force method demo for the Egg Dropping game.

This script imports `EggGame` and provides a `main()` that
demonstrates a simple linear search to identify the breaking point.
"""
from typing import Tuple
from egg_game import EggGame


def brute_force_find_breaking_point(game: EggGame, start_floor=0) -> Tuple[int, int]:
    """
    Linearly scan floors from 0 upward and drop an egg at each floor
    until an egg breaks. The first breaking floor is the breaking point.

    If no floor causes a break or other issues happen, return -1
    """
    n_floors = game.get_nbr_floors()
    for floor in range(start_floor, n_floors):
        result = game.drop_egg(floor)
        if result == 1:
            # First break (1) indicates this floor is the breaking point.
            _, _, _, drop_cnt = game.get_state()
            return floor, drop_cnt
        elif result == 2 or result == 3:
            return -1, 0
    # No break occurred => breaking point equals number of floors
    return -1, 0

def sampling_find_breaking_point(game: EggGame, sampling_points=[]) -> Tuple[int, int]:
    n_floors = game.get_nbr_floors()

    if sampling_points == []:
        sampling_points = list(range(n_floors+1)) # default sampling every floor (like brute force)

    # sample till first egg breaks
    last_non_breaking_sampling_point = sampling_points[0]
    for sampling_point in sampling_points:
        result = game.drop_egg(sampling_point)
        if result == 1:
            # First egg breaks
            break
        if result == 0:
            # Egg did not break -> continue sampling
            last_non_breaking_sampling_point = sampling_point
        elif result == 2 or result == 3:
            # Error
            return -1, 0

    # brute force search from last valid sampling point
    return brute_force_find_breaking_point(game, start_floor=last_non_breaking_sampling_point)




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

    game2 = EggGame(n_eggs=n_eggs, n_floors=n_floors, breaking_pnt=breaking_pnt)
    found_breaking_point = brute_force_find_breaking_point(game2, start_floor=19)

    # Single allowed guess: verify our found breaking point
    is_correct = game2.guess_breaking_pnt(found_breaking_point)

    print(
        f"Found breaking point: {found_breaking_point} | "
        f"Correct: {is_correct} | State: {game2.get_state()}"
    )


if __name__ == "__main__":
    main()
