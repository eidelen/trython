from egg_game import EggGame

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_game = EggGame(2, 10, 6)
    print(test_game.get_state())
    print(test_game.drop_egg(0))
    print(test_game.drop_egg(5))
    print(test_game.drop_egg(11))
    print(test_game.drop_egg(6))
    print(test_game.drop_egg(8))
    print(test_game.drop_egg(8))
    print(test_game.get_state())
    print(test_game.guess_breaking_pnt(6))
    try:
        print(test_game.guess_breaking_pnt(5))
    except RuntimeError as e:
        print(f"Second guess not allowed: {e}")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
