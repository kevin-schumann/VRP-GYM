def test(test_str):
    import random

    print(test_str.format(test=random.random()))


for i in range(10):
    test("{test}")
