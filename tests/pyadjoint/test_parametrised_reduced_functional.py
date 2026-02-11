from pyadjoint import *
from pyadjoint.reduced_functional import ParametrisedReducedFunctional, ReducedFunctional
from pyadjoint.placeholder import Placeholder

def test_prf():
    a = AdjFloat(2.0)
    b = AdjFloat(3.0)
    c = a*b
    d = AdjFloat(5.0)

    e = c*d
    Jhat = ParametrisedReducedFunctional(e, Control(a), d)
    assert Jhat(2.0) == e

    Jhat.parameter_update(10.0)
    assert Jhat(2.0) == e * 2
    assert Jhat.derivative()[0] == b*10.0
