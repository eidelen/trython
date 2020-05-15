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

def get_reward_vector() -> np.ndarray:
    r = np.array( [0, -1, -1,  -1, -1, -1,  -1, -1, 0 ] ).transpose()
    return r


class MyTestCase(unittest.TestCase):

    def test_value_iteration(self):
        #   x  -1 -2
        #   -1 -2 -1
        #   -2 -1 -x

        p = get_state_transition_matrix()
        v = np.zeros( (9) )
        r = get_reward_vector()
        for k in range(0,5):
            v_n = v.copy()
            for s in range(0, 9):
                ps = p[s][:].transpose()
                future_value = ps * v
                all_v = r + future_value
                u = 2








        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
