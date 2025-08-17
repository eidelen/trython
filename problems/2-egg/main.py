from egg_game import EggGame
from methods import brute_force_find_breaking_point, sampling_find_breaking_point, generate_random_sampling_points
from benchmark import benchmark_egg_methods

def test_egg_methods():
    benchmark_egg_methods(("BruteForce", brute_force_find_breaking_point, None))
    benchmark_egg_methods(("Sampling All", sampling_find_breaking_point, None))
    benchmark_egg_methods(("Every Second", sampling_find_breaking_point, [10, 20, 30, 40, 50, 60, 70, 80, 90]))

def find_best_sampling_interval():
    n_floors = 100
    for step_size in range(1, n_floors):
        sampling_points = []
        for i in range(n_floors):
            sampling_point = i * step_size
            if sampling_point < n_floors:
                sampling_points.append(sampling_point)
            else:
                break

        benchmark_egg_methods(("Interval = " + str(step_size), sampling_find_breaking_point, sampling_points))


def do_permutation(n_floors, n_eggs, current_floor, sampling_points):

    global very_best_solution

    # add no further sampling points
    best_solution = benchmark_egg_methods(n_floors, n_eggs, ("Sampling Method", sampling_find_breaking_point, sampling_points))
    if best_solution < very_best_solution:
        very_best_solution = best_solution
        print("Best solution found! ", very_best_solution, sampling_points)


    # if last sampling point not already max, continue branching
    next_floor = current_floor + 1
    if next_floor < n_floors:
        for i in range(next_floor, n_floors):
            new_sampling_points = sampling_points.copy()
            new_sampling_points.append(i)

            # progress
            #if current_floor < 90:
            #    print("Progress: ", sampling_points)

            avg_drops = do_permutation(n_floors, n_eggs, i, new_sampling_points)
            if avg_drops < best_solution:
                best_solution = avg_drops
                if best_solution < very_best_solution:
                    very_best_solution = best_solution
                    print("Best solution found! ", very_best_solution, sampling_points)

    return best_solution


def find_best_sampling():
    global very_best_solution
    n_floors = 30
    n_eggs = 2
    very_best_solution = n_floors
    do_permutation(n_floors, n_eggs, 0, [])



def find_best_random_sampling():
    n_floors = 100
    n_eggs = 2
    n_samples_per_epoch = 100000
    base = []
    variation = 10
    best_solution = n_floors
    for e in range(100):
        print("Epoch ", e)
        best_sampling_points_in_epoch = []
        for i in range(n_samples_per_epoch):
            sampling_points = generate_random_sampling_points(n_floors, base, variation)
            this_solution = benchmark_egg_methods(n_floors, n_eggs, ("Random sampling Method", sampling_find_breaking_point, sampling_points))

            if this_solution < best_solution:
                best_solution = this_solution
                best_sampling_points_in_epoch = sampling_points
                print("Best solution found! ", best_solution, sampling_points)

        base = best_sampling_points_in_epoch
















# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # find_best_sampling_interval()
    #test_egg_methods()
    #game = EggGame(n_eggs=2, n_floors=100, breaking_pnt=0)
    #comp_breaking_pnt, drop_cnt = sampling_find_breaking_point(game)
    #print(comp_breaking_pnt, drop_cnt)

    # find_best_sampling()

    #print(benchmark_egg_methods(("9", sampling_find_breaking_point, [3, 5, 7])))

    find_best_random_sampling()

    # optimal
    #print(benchmark_egg_methods(100, 2, ("opt", sampling_find_breaking_point, [13, 26, 38, 49, 59, 68, 76, 83, 89, 94, 98])))




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
