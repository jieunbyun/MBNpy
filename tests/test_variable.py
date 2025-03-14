import numpy as np
import pytest

from mbnpy import variable

np.set_printoptions(precision=3)

def test_init1():
    name = 'A'
    value = ['failure', 'survival']

    var = {'name': name, 'values': value}
    a = variable.Variable(**var)

    assert isinstance(a, variable.Variable)
    np.testing.assert_array_equal(a.name, name)
    np.testing.assert_array_equal(a.values, value)
    np.testing.assert_array_equal(a.B_flag, 'auto')
    np.testing.assert_array_equal(a.B, [{0}, {1}, {0, 1}])

def test_init2():
    name = 'A'
    a = variable.Variable(name)
    value = ['failure', 'survival']
    a.values = value

    assert isinstance(a, variable.Variable)
    np.testing.assert_array_equal(a.name, name)
    np.testing.assert_array_equal(a.values, value)
    np.testing.assert_array_equal(a.B_flag, 'auto')
    np.testing.assert_array_equal(a.B, [{0}, {1}, {0, 1}])

def test_init3():
    name = 'A'
    a = variable.Variable(name, B_flag='fly')
    value = ['failure', 'survival']
    a.values = value

    assert isinstance(a, variable.Variable)
    np.testing.assert_array_equal(a.name, name)
    np.testing.assert_array_equal(a.values, value)
    np.testing.assert_array_equal(a.B_flag, 'fly')
    assert a.B is None

def test_eq1():
    name = 'A'
    value = ['failure', 'survival']

    var = {'name': name, 'values': value}
    a = variable.Variable(**var)
    b = variable.Variable(**var)

    assert a == b

def test_eq2():
    var1 = {'name': 'A', 'values': ['failure', 'survival']}
    var2 = {'name': 'B', 'values': [0, 1, 2]}
    a = variable.Variable(**var1)

    b = variable.Variable(**var1)
    c = variable.Variable(**var2)
    _list = [b, c]

    assert a in _list

def test_get_state1():
    varis = {'x1': variable.Variable(name='x1', values=[0,1]),
             'x2': variable.Variable('x2', [0,1,2,3]),
             'x3': variable.Variable('x3', np.arange(20).tolist())}
    
    assert varis['x1'].get_state({0,1}) == 2
    assert varis['x2'].get_state({0,1,2}) == 10
    assert varis['x3'].get_state({3,4,5}) == 670

def test_get_set1():
    varis = {'x1': variable.Variable('x1', [0,1]),
             'x2': variable.Variable('x2', [0,1,2,3]),
             'x3': variable.Variable('x3', np.arange(20).tolist())}
    
    assert varis['x1'].get_set( 2 ) == {0,1}
    assert varis['x2'].get_set( 10 ) == {0,1,2}
    assert varis['x3'].get_set( 670 ) == {3,4,5}

def test_get_state_from_vector1():
    varis = {'x1': variable.Variable('x1', [0,1]),
             'x2': variable.Variable('x2', [0,1,2,3]),
             'x3': variable.Variable('x3', np.arange(20).tolist())}
    
    assert varis['x1'].get_state_from_vector([1,1]) == 2
    assert varis['x2'].get_state_from_vector([1,1,1,0]) == 10
    assert varis['x3'].get_state_from_vector([0,0,0,1,1,
                                              1,0,0,0,0,
                                              0,0,0,0,0,
                                              0,0,0,0,0]) == 670
    
def test_get_state_from_vector2():
    varis = {'x1': variable.Variable('x1', [0,1,2])}
    assert varis['x1'].get_state_from_vector([0,0,0]) == -1

def test_get_state_from_vector3():
    varis = {'x1': variable.Variable('x1', [0,1,2])}
    with pytest.raises(AssertionError):
        varis['x1'].get_state_from_vector([1,1])

def test_get_Bst_from_Bvec1():

    varis = {'x1': variable.Variable( 'x1', [0,1,2])}

    Bvec = np.array([[[1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],                        
                     [[1, 0, 0],
                      [1, 0, 0],
                      [1, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]],                        
                     [[0, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]],                      
                     [[0, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]],
                     [[0, 1, 1],
                      [0, 0, 0],
                      [0, 0, 0],
                      [0, 1, 0],
                      [0, 1, 0],
                      [0, 0, 1]]])
    
    Bst = varis['x1'].get_Bst_from_Bvec( Bvec )
    np.testing.assert_array_equal(Bst, np.array([[0, 0, 0, -1, -1, -1],
                                                 [0, 0, 0, -1, -1, -1],
                                                 [5, -1, -1, 1, 1, 2],
                                                 [5, -1, -1, 1, 1, 2],
                                                 [5, -1, -1, 1, 1, 2]]))