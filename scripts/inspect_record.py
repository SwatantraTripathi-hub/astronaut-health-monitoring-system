import sys
from collections import Counter
import wfdb

path = sys.argv[1]            # e.g. data/raw/mitdb/100
rec = wfdb.rdrecord(path)
print("fs:", rec.fs, "| samples:", rec.sig_len,
      "| duration (s):", rec.sig_len / rec.fs)
print("signals:", rec.sig_name, rec.units)
print("comments:", rec.comments)
