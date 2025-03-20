def convert_from_string(value: str):
    """Tries to convert a string to the most appropriate type. Leaves it unchanged if conversion does not succeed.

    Note that booleans use the Fortran syntax and real numbers in double precision can use the "old" Fortran `D`
    delimiter.
    """
    # Start by trying to convert from a Fortran logical to a Python bool
    if value.lower() == ".true.":
        return True
    elif value.lower() == ".false.":
        return False
    # Next try to convert to integer or float
    for conversion in [
        lambda: int(value),
        lambda: float(value),
        lambda: float(value.replace("D", "e")),
    ]:
        try:
            out = conversion()
        except ValueError:
            continue
        return out
    # None of the above succeeded, so just return the string
    return value


def convert_to_string(value) -> str:
    """Convert a value to a string.

    Note that booleans are converted using the Fortran syntax and real numbers in double precision use the "old" Fortran
     `D` delimiter for backward compatibility.
    """
    if isinstance(value, bool):
        return ".true." if value else ".false."
    elif isinstance(value, float):
        return "{:e}".format(value).replace("e", "D")
    else:
        return str(value)


def nano_to_sec(nanos):
    """Convert nanoseconds to seconds."""
    return nanos / (1000 * 1000 * 1000)
