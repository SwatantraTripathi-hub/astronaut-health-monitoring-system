import wfdb

wfdb.dl_files("mitdb", "data/raw/mitdb",
              ["100.dat", "100.hea", "100.atr",
               "201.dat", "201.hea", "201.atr",
               "202.dat", "202.hea", "202.atr"])

wfdb.dl_files("bidmc", "data/raw/bidmc",
              ["bidmc01.dat", "bidmc01.hea",
               "bidmc01n.dat", "bidmc01n.hea",
               "bidmc01.breath", "LICENSE"])
