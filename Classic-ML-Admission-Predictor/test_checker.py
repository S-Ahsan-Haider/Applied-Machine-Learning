from dummy import get_square

def test1():
    x = 5
    r = get_square(x)
    assert r == 25

def test2():
    x = -3
    r = get_square(x)
    assert r == 9