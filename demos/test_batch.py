import pytest


from demos.pf_map import batch as pf_map
from demos.rbd import batch as rbd
from demos.power_house import batch as power_house
from demos.routine import batch as routine
from demos.road import batch as road
from demos.SF import batch as sf


def test_batch_rbd():

    rbd.main()


def test_batch_pf():
    pf_map.debug()


def test_batch_routine():
    routine.main()


def test_batch_road():
    road.main()


def test_batch_sf():
    sf.main(max_sf=10)


def test_batch_power():
    power_house.batch()
