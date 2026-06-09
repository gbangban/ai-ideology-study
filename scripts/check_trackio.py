import trackio
print("version:", getattr(trackio, "__version__", "unknown"))
print("file:", trackio.__file__)
print("dir:", [x for x in dir(trackio) if not x.startswith("_")])
