"""Flag verbatim word-sequence overlap between this repo's text and a source document you must not copy.

    python tools/check_text_overlap.py path/to/paper.txt [--n 6]

Extract a PDF first (e.g. `pdftotext paper.pdf paper.txt`). Exit code 1 if any N-word sequence is shared.
This catches verbatim copying only; paraphrase closeness still needs human judgement.
"""
import argparse, glob, json, os, re, sys

def norm(t): return re.findall(r"[a-z0-9']+", t.lower())

def text_of(path):
    if path.endswith(".ipynb"):
        return "\n".join("".join(c["source"]) for c in json.load(open(path, encoding="utf-8"))["cells"])
    return open(path, encoding="utf-8", errors="ignore").read()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("source"); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    src = norm(open(a.source, encoding="utf-8", errors="ignore").read())
    shingles = {tuple(src[i:i + a.n]) for i in range(len(src) - a.n + 1)}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); bad = 0
    for f in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if not os.path.isfile(f) or not f.endswith((".py", ".md", ".yml", ".ipynb", ".toml", ".txt")) or "/data/" in f:
            continue
        w = norm(text_of(f)); hits = {" ".join(w[i:i + a.n]) for i in range(len(w) - a.n + 1) if tuple(w[i:i + a.n]) in shingles}
        if hits:
            bad += len(hits); print(f"{os.path.relpath(f, root)}: {len(hits)} shared sequence(s), e.g. '{sorted(hits)[0]}'")
    print("OK: no shared sequences" if not bad else f"FOUND {bad} shared sequence(s)")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
