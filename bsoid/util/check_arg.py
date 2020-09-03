import bsoid


def ensure_type(obj, expected_type):
    """"""
    if not isinstance(obj, expected_type):
        type_err = f'For object (value = {obj}), expected type was {expected_type} but instead found {type(obj)}'
        raise TypeError(type_err)