import os
from pathlib import Path

if "PRISM_DATADRIVE" in os.environ:
    fp = os.environ["PRISM_DATADRIVE"]
else:
    raise ValueError("PRISM_DATADRIVE environment variable not set")
datadrive = Path(fp)