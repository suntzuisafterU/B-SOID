import bsoid





def ensure_type(var, expected_type):
    """"""
    if not isinstance(var, expected_type):
        type_err = f'For object (value = {var}), expected type was {expected_type} but instead found {type(var)}'
        raise TypeError(type_err)


def ensure_collection_not_empty(collection):

    if len(collection) == 0:
        err = f''
        raise ValueError(err)


