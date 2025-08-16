from egg_game import EggGame
from methods import brute_force_find_breaking_point, sampling_find_breaking_point
from benchmark import benchmark_egg_methods

def test_egg_methods():
    benchmark_egg_methods( ("BruteForce", brute_force_find_breaking_point, None) )
    benchmark_egg_methods(("Sampling All", sampling_find_breaking_point, None))
    benchmark_egg_methods(("Every Second", sampling_find_breaking_point, [10, 20, 30, 40, 50, 60, 70, 80, 90]))








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_egg_methods()
    #game = EggGame(n_eggs=2, n_floors=100, breaking_pnt=0)
    #comp_breaking_pnt, drop_cnt = sampling_find_breaking_point(game)
    #print(comp_breaking_pnt, drop_cnt)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
