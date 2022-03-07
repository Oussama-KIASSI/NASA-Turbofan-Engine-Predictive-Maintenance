import pandas as _pd
import os as _os
import re as _re


def data_loader(dataprefix1: str,
                dataschema1: list[str],
                dataschema2: list[str],
                sourcepath: str = '../data',
                datatype: str = '01_raw',
                dataext: str = '.txt',
                sep: str = ' '):
    """

    Args:
        dataprefix1:
        dataschema1:
        dataschema2:
        sourcepath:
        datatype:
        dataext:
        sep:

    Returns:

    """
    datapath = _os.path.join(sourcepath, datatype)
    files = {}
    for file in _os.listdir(datapath):
        if file.endswith(dataext):
            filepath = _os.path.join(datapath, file)
            filename = _re.sub(dataext, '', file)
            files[filename] = filepath
            if file.startswith(dataprefix1):
                globals()[filename] = _pd.read_csv(filepath, sep=sep, names=dataschema1, usecols=dataschema1)
            else:
                globals()[filename] = _pd.read_csv(filepath, sep=sep, names=dataschema2, usecols=dataschema2)
    return files
