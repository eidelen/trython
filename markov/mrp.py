import unittest
import numpy as np

# markov reward process

#  x: the goal   -: field to move    actions: u/d/l/r    reward: -1 for each move
#   x - -
#   - - -
#   - - x

#  numeric state values
#   0 1 2
#   3 4 5
#   6 7 8

def get_state_transition_matrix() -> np.ndarray:
    p = np.array( [ [1.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25,  0.0, 0.25, 0.0,  0.0, 0.0, 0.0],
                    [0.0, 0.25, 0.50,  0.0, 0.0, 0.25,  0.0, 0.0, 0.0 ],
                    [0.25, 0.0, 0.0,  0.25, 0.25, 0.0,  0.25, 0.0, 0.0],
                    [0.0, 0.25, 0.0,  0.25, 0.0, 0.25,  0.0, 0.25, 0.0],
                    [0.0, 0.0, 0.25,  0.0, 0.25, 0.25,  0.0, 0.0, 0.25],
                    [0.0, 0.0, 0.0,  0.25, 0.0, 0.0,  0.5, 0.25, 0.0],
                    [0.0, 0.0, 0.0,  0.0, 0.25, 0.0,  0.25, 0.25, 0.25],
                    [0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0]])
    return p

def get_goals() -> np.ndarray:
    g = np.array( [0 ] ).transpose()
    return g


class MyTestCase(unittest.TestCase):

    def test_value_iteration(self):
        #   x  -1 -2
        #   -1 -2 -1
        #   -2 -1 -x

        p = get_state_transition_matrix()
        v = np.zeros( (9) )
        goals = get_goals()
        for k in range(0,10):
            v_n = v.copy()
            for s in range(0, 9):
                if s in goals:
                    v_n[s] = 0
                else:
                    states_after_actions = np.where(p[s] > 0.0) # check which states are reachable from state s
                    best_value = -100000
                    for ns in np.nditer(states_after_actions):
                        # action leads to one specific new state -> probability 1.0.
                        # -1 for reward
                        value_guess = v[ns] - 1
                        if best_value < value_guess:
                            best_value = value_guess
                    v_n[s] = best_value
            v = v_n.copy()

            print("Value iteration: ", k)
            for m in range(0, 3):
                print( v[3*m:3*m+3] )


        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
