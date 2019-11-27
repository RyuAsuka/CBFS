def time_formatter(seconds):
    """
    Return a simpler formatter of a big number of seconds.

    Parameters
    ----------
        seconds : float
            The number of seconds, may be very big.

    Returns
    -------
        str :
            A string with format <hour: minute: seconds>
    """
    if seconds < 60.0:
        return str(seconds)
    else:
        minute = 0
        while seconds > 60.0:
            minute += 1
            seconds -= 60.0
        if minute < 60:
            return f'{minute}:{seconds}'
        else:
            hour = 0
            while minute > 60:
                hour += 1
                minute -= 60
            return f'{hour}:{minute}:{seconds}'
