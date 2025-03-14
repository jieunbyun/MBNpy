import pf_map.batch as pf_map
import rbd.batch as rbd
import power_house.batch as power_house
import routine.batch as routine
import road.batch as road
import SF.batch as sf

def test_batch():

    rbd.main()
    pf_map.debug()
    routine.main()
    road.main()
    sf.main(max_sf=10)
    #power_house.batch()
