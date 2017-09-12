# -*- encoding: utf-8 -*-

"""
Miscellaneous utility functions.
"""

from array import array


def array_double(values):
    """
    Returns a compact array-like representation of double values.
    """
    return array('d', values)


def array_int(values):
    """
    Returns a compact array-like representation of integer values.
    """
    return array('l', values)


def array_bool(values):
    """
    Returns a compact array-like representation of boolean values in a
    given iterable.
    """
    return array('B', values)


def mean(sequence):
    return sum(sequence) / max(len(sequence), 1.0)


def median(lst):
    """
    Returns median of the lst.
    """
    sorted_list = sorted(lst)
    list_len = len(sorted_list)
    index = (list_len - 1) // 2
    if list_len % 2:
        return sorted_list[index]
    else:
        return (sorted_list[index] + sorted_list[index + 1])/2.0
