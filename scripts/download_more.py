import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/bidmc/1.0.0/"
target = Path("data/raw/bidmc")
target.mkdir(parents=True, exist_ok=True)

for n in range(2, 6):
    for suffix in (".dat", ".hea", "n.dat", "n.hea", ".breath"):
        name = f"bidmc{n:02d}{suffix}"
        dest = target / name
        if dest.exists():
            continue
        urllib.request.urlretrieve(BASE + name, dest)
        print("got", name)
