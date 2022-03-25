import math

import numpy as np


def channel(message:str, mode=None, ber=None, errasures=None) -> str:
    """Simulate a channel where bits are flipped. 
    The number of bits flipper are conform the ber.

    
    Arguments:
        message {str} -- Bit string 
    
    Keyword Arguments:
        mode {str} -- burst mode or otherwise locations are random (default: {None})
        ber {float} -- Ratio number of bits flipped in % (default: {None})
        errasures {float} -- Ratio of bits errased in %(default: {None})
    """

    assert mode is None, "Mode is not supported"
    assert errasures is None, "errasures is not supported"

    
    num_locations = math.floor((ber/100)*len(message))
    locations = np.arange(len(message))
    np.random.shuffle(locations)
    locations_flipped = locations[:num_locations].tolist()
    message = list(message)
    print("\t\t generating {} flips from a total of {} elements".format(num_locations, len(message)))
    
    for location in locations_flipped:
        message[location] = "0" if message[location] is "1" else "1"

    return message



