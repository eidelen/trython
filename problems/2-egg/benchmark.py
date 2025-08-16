from egg_game import EggGame
from methods import brute_force_find_breaking_point

def benchmark_egg_methods(method):
    nbr_floors = 100
    nbr_eggs = 2

    n_experiements = 0
    avg_drops = 0

    method_name, method_call, additional_arg = method

    print("Asses method ", method_name)
    for breaking_pnt in range(nbr_floors):
        game = EggGame(n_eggs=nbr_eggs, n_floors=nbr_floors, breaking_pnt=breaking_pnt)
        if additional_arg is not None:
            comp_breaking_pnt, drop_cnt = method_call(game, additional_arg)
        else:
            comp_breaking_pnt, drop_cnt = method_call(game)

        if comp_breaking_pnt < 0:
            print("Error in method")
            break
        n_experiements += 1
        avg_drops += drop_cnt

    avg_drops /= n_experiements
    print("Average drops: ", avg_drops)
    return avg_drops
