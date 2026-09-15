"""TODO: add this docstring"""



from importlib.metadata import PackageNotFoundError, version



try:
    __version__ = version("slodl")
except PackageNotFoundError:
    __version__ = "0+unknown"
