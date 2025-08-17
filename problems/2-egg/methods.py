"""
Brute-force method demo for the Egg Dropping game.

This script imports `EggGame` and provides a `main()` that
demonstrates a simple linear search to identify the breaking point.
"""
from typing import Tuple, List
from random import randint
from math import ceil
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
    last_non_breaking_sampling_point = 0
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


def generate_random_sampling_points(n_floors: int, base=[], variation = 2) -> List[int]:
    sampling_points = []
    last_sampling_point = -1
    if base == []:
        while last_sampling_point + 1 < n_floors:
            next_sampling_point = randint(last_sampling_point + 1, n_floors - 1)
            sampling_points.append(next_sampling_point)
            last_sampling_point = next_sampling_point
    else:
        for b in base:
            offset = randint(-variation, +variation)
            next_sampling_point = b + offset
            if next_sampling_point + 1 < n_floors and next_sampling_point > last_sampling_point:
                sampling_points.append(next_sampling_point)
                last_sampling_point = next_sampling_point

    # randomly remove or add entries
    final_sampling_points = []
    last_sampling_point = 0
    for s in sampling_points:
        r = randint(0, 10)
        if r < 3:
            #remove it
            pass
        elif r < 6:
            # add an intermediate point
            intermediate_sampling_point = ceil((last_sampling_point + s) / 2)
            if last_sampling_point < intermediate_sampling_point < s:
                final_sampling_points.append(intermediate_sampling_point)
            final_sampling_points.append(s)
            last_sampling_point = s
        else:
            final_sampling_points.append(s)
            last_sampling_point = s

    return final_sampling_points



def main():
    # Example configuration; adjust as needed
    n_eggs = 2
    n_floors = 100
    breaking_pnt = 20

    game = EggGame(n_eggs=n_eggs, n_floors=n_floors, breaking_pnt=breaking_pnt)
    found_breaking_point, attempts = brute_force_find_breaking_point(game)

    # Single allowed guess: verify our found breaking point
    is_correct = game.guess_breaking_pnt(found_breaking_point)

    print(
        f"BF Found breaking point: {found_breaking_point} | "
        f"BF Correct: {is_correct} | State: {game.get_state()}"
    )

    game2 = EggGame(n_eggs=n_eggs, n_floors=n_floors, breaking_pnt=breaking_pnt)
    found_breaking_point, attempts = brute_force_find_breaking_point(game2, start_floor=19)

    # Single allowed guess: verify our found breaking point
    is_correct = game2.guess_breaking_pnt(found_breaking_point)

    print(
        f"BF Found breaking point: {found_breaking_point} | "
        f"BF Correct: {is_correct} | State: {game2.get_state()}"
    )

    game3 = EggGame(n_eggs=n_eggs, n_floors=n_floors, breaking_pnt=breaking_pnt)
    found_breaking_point, attempts = sampling_find_breaking_point(game3)

    # Single allowed guess: verify our found breaking point
    is_correct = game3.guess_breaking_pnt(found_breaking_point)

    print(
        f"SAMP Found breaking point: {found_breaking_point} | "
        f"SAMP Correct: {is_correct} | State: {game3.get_state()}"
    )

    game4 = EggGame(n_eggs=n_eggs, n_floors=n_floors, breaking_pnt=breaking_pnt)
    found_breaking_point, attempts = sampling_find_breaking_point(game4, [19, 21])

    # Single allowed guess: verify our found breaking point
    is_correct = game4.guess_breaking_pnt(found_breaking_point)

    print(
        f"SAMP Found breaking point: {found_breaking_point} | "
        f"SAMP Correct: {is_correct} | State: {game4.get_state()}"
    )

    for i in range(50):
        rnd_samp = generate_random_sampling_points(100, [10, 20, 30], 1)
        print(rnd_samp)



if __name__ == "__main__":
    main()
