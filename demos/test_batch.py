import pytest


from demos.pf_map import batch as pf_map
from demos.rbd import batch as rbd
from demos.power_house import batch as power_house
from demos.routine import batch as routine
from demos.road import batch as road
from demos.SF import batch as sf


@pytest.mark.skip('get_composite_variable')
def test_batch():

    rbd.main()
    pf_map.debug()
    routine.main()
    road.main()
    sf.main(max_sf=10)
    #power_house.batch()
