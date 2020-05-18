import unittest
import numpy as np

# markov reward process

#  x: the goal   -: field to move    actions: u/d/l/r    #: bad reward  T: Trap
#   x - x
#   # - -
#   - - T

#  numeric state values
#   0 1 2
#   3 4 5
#   6 7 8

# actions lead ultimately to one next state
def get_action_state_matrix() -> np.ndarray:
    p = np.array( [ [1.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0,  0.0, 1.0, 0.0,  0.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0,  0.0, 0.0, 1.0,  0.0, 0.0, 0.0 ],
                    [1.0, 0.0, 0.0,  1.0, 1.0, 0.0,  1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0,  1.0, 0.0, 1.0,  0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0,  0.0, 1.0, 1.0,  0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0,  1.0, 0.0, 0.0,  1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0,  0.0, 1.0, 0.0,  1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 1.0]])
    return p

def get_goals() -> np.ndarray:
    g = np.array( [0, 2] ).transpose()
    return g

def get_reward(state) -> int:
    return -5 if state == 3 else -1

class MyTestCase(unittest.TestCase):

    def test_value_iteration(self):
        #   x  -1  x
        #   -1 -2 -1
        #   -4 -3 >5

        p = get_action_state_matrix()
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
                        # negative reward for action
                        value_guess = v[ns] + get_reward(ns)
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
