import time

import numpy as np


def uint8_to_bit(uint8_list):
    return uintx_to_bit(uint8_list, 8)


def uint16_to_bit(uint16_list):
    return uintx_to_bit(uint16_list, 16)


def uint32_to_bit(uint32_list):
    return uintx_to_bit(uint32_list, 32)


def uintx_to_bit(uintx_list, width=8):
    return "".join([np.binary_repr(x, width=width) for x in uintx_list])


# def bit_to_uint8(bit_list):
#     """Converts a bit string to a numpy uint8 array
#
#     Arguments:
#         bit_list {str} -- Bit list expecting string, otherwise the list is first converted to a btt string
#
#     Returns:
#         np.ndarray -- uint8 typed ndarray
#     """
#
#     if type(bit_list) is not str:
#         bit_list = "".join([x for x in bit_list])
#
#     assert len(bit_list) % 8 == 0, "Provided bits length should be divisible by 8"
#
#     chunked_list = [chunk for chunk in _chunks(bit_list, 8)]
#
#     return np.array([int(bits, 2) for bits in chunked_list], dtype=np.uint8)

def bit_to_uint8(bit_list):
    return bit_to_uintx(bit_list, width=8)


def bit_to_uint16(bit_list):
    return bit_to_uintx(bit_list, width=16)


def bit_to_uint32(bit_list):
    return bit_to_uintx(bit_list, width=32)


def bit_to_uintx(bit_list, width=8):
    """Converts a bit string to a numpy uint8 array

    Arguments:
        bit_list {str} -- Bit list expecting string, otherwise the list is first converted to a btt string

    Returns:
        np.ndarray -- uint8 typed ndarray
    """

    def _chunks(s, n):
        """Produce `n`-character chunks from `s`."""
        for start in range(0, len(s), n):
            yield s[start:start + n]

    if width == 8:
        _type = np.uint8
    elif width == 16:
        _type = np.uint16
    elif width == 32:
        _type = np.uint32
    elif width == 64:
        _type = np.uint64
    else:
        ValueError(f"Width ({width}) not supported")

    if type(bit_list) is not str or type(bit_list[0]) is not str:
        bit_list = "".join([str(x) for x in bit_list])

    assert len(bit_list) % width == 0, f"Provided bits length should be divisible by {width}"

    uintx_list = np.array([int(chunk, 2) for chunk in _chunks(bit_list, width)], dtype=_type)

    return uintx_list.flatten()


# def bit_to_uint8_fast(bit_list):
#     """Converts a bit string to a numpy uint8 array
#
#     Arguments:
#         bit_list {str} -- Bit list expecting string, otherwise the list is first converted to a btt string
#
#     Returns:
#         np.ndarray -- uint8 typed ndarray
#     """
#
#     if type(bit_list) is str:
#         bit_list = np.array([int(x) for x in bit_list])
#
#     assert len(bit_list) % 8 == 0, "Provided bits length should be divisible by 8"
#
#     bit_matrix = np.resize(bit_list, (-1, 8))
#
#     uint8_list = np.packbits(bit_matrix)
#
#     return uint8_list


class Time:
    def __init__(self):
        self.t = None

    def tic(self):
        self.t = time.time()
        return self.t

    def toc(self, t=None):
        if t is None:
            assert self.t is not None, "Call tic() before toc()"
            diff = time.time() - self.t
            self.t = None
        else:
            diff = time.time() - t
        return diff

    def toc_str(self):
        time_diff = self.toc()

        if time_diff < 1:
            time_diff *= 1000
            unit = "ms"
        elif time_diff < 100:
            unit = "s"
        else:
            time_diff /= 60
            unit = "min"

        return f"{time_diff:.2f} {unit}"

    def toc_print(self):
        print(self.toc_str())
