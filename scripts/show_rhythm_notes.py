import sys
import wfdb


def main():
    record = sys.argv[1] if len(sys.argv) > 1 else "100"
    ann = wfdb.rdann(f"data/raw/mitdb/{record}", "atr")
    for i, sym in enumerate(ann.symbol):
        if sym == "+":
            print(f"{ann.sample[i] / ann.fs if hasattr(ann, 'fs') else ann.sample[i] / 360} {ann.aux_note[i] if i < len(ann.aux_note) else ''}")


if __name__ == "__main__":
    main()
