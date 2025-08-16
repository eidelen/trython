import unittest
from egg_game import EggGame


class TestEggGame(unittest.TestCase):

    def test_initial_state(self):
        game = EggGame(n_eggs=2, n_floors=10, breaking_pnt=6)
        self.assertEqual(game.get_state(), (10, 6, 2, 0))

    def test_drop_invalid_floor(self):
        game = EggGame(n_eggs=2, n_floors=10, breaking_pnt=6)
        self.assertEqual(game.drop_egg(-1), 3)
        self.assertEqual(game.drop_egg(10), 3)
        # No change in state for invalid drops
        self.assertEqual(game.get_state(), (10, 6, 2, 0))

    def test_safe_floor_no_break(self):
        game = EggGame(n_eggs=2, n_floors=10, breaking_pnt=6)
        # Floor 5 < breaking point 6 => safe
        self.assertEqual(game.drop_egg(5), 0)
        n_floors, breaking_pnt, eggs_left, drops = game.get_state()
        self.assertEqual(eggs_left, 2)
        self.assertEqual(drops, 1)

    def test_breaking_floor_egg_breaks(self):
        game = EggGame(n_eggs=2, n_floors=10, breaking_pnt=6)
        # Floor 6 >= breaking point 6 => breaks
        self.assertEqual(game.drop_egg(6), 1)
        n_floors, breaking_pnt, eggs_left, drops = game.get_state()
        self.assertEqual(eggs_left, 1)
        self.assertEqual(drops, 1)

        # Break the last egg
        self.assertEqual(game.drop_egg(7), 1)
        # Then game over
        self.assertEqual(game.drop_egg(7), 2)
        n_floors, breaking_pnt, eggs_left, drops = game.get_state()
        self.assertEqual(eggs_left, 0)
        self.assertEqual(drops, 2)

    def test_game_over_no_more_drops(self):
        game = EggGame(n_eggs=1, n_floors=10, breaking_pnt=2)
        # Game over when trying to drop an egg even though there is non left
        self.assertEqual(game.drop_egg(2), 1)
        # Further attempts return 2 and do not increase drop count
        self.assertEqual(game.drop_egg(0), 2)
        self.assertEqual(game.get_state(), (10, 2, 0, 1))

    def test_guess_breaking_point(self):
        game = EggGame(n_eggs=2, n_floors=10, breaking_pnt=6)
        # First guess allowed and correct
        self.assertTrue(game.guess_breaking_pnt(6))
        # Second guess should not be allowed
        with self.assertRaises(RuntimeError):
            game.guess_breaking_pnt(5)

    def test_guess_only_once_even_if_wrong(self):
        game = EggGame(n_eggs=2, n_floors=10, breaking_pnt=6)
        # First guess wrong still consumes the guess
        self.assertFalse(game.guess_breaking_pnt(5))
        with self.assertRaises(RuntimeError):
            game.guess_breaking_pnt(6)


if __name__ == '__main__':
    unittest.main()
