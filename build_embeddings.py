"""Train word embeddings on Ulysses itself: PPMI over a +-5 word window,
then truncated SVD. Classic distributional semantics, no neural net.
Export vectors only for the words the site's paragraph needs (plus the
visual-anchor seed words), as compact JSON.
"""
import re, json, sys, collections
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import svds

text = ""
for path in sys.argv[1].split(","):
    RAW = open(path, encoding="utf-8-sig").read()
    start = RAW.index("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = RAW.index("\n", start) + 1
    end = RAW.index("*** END OF THE PROJECT GUTENBERG EBOOK")
    text += RAW[start:end].lower() + "\n"

tokens = re.findall(r"[a-zà-ÿ]+(?:['’][a-zà-ÿ]+)*", text)
print(f"tokens: {len(tokens):,}", file=sys.stderr)

VOCAB_N = 14000
WINDOW = 5
DIM = 48

counts = collections.Counter(tokens)
vocab = [w for w, c in counts.most_common(VOCAB_N)]
wid = {w: i for i, w in enumerate(vocab)}

# co-occurrence with distance weighting 1/d
C = lil_matrix((VOCAB_N, VOCAB_N), dtype=np.float32)
ids = [wid.get(t, -1) for t in tokens]
for i, a in enumerate(ids):
    if a < 0: continue
    for d in range(1, WINDOW + 1):
        j = i + d
        if j >= len(ids): break
        b = ids[j]
        if b < 0: continue
        w = 1.0 / d
        C[a, b] += w
        C[b, a] += w
C = csr_matrix(C)
print(f"cooc nnz: {C.nnz:,}", file=sys.stderr)

# PPMI with context-distribution smoothing (alpha=.75)
row = np.asarray(C.sum(axis=1)).ravel()
col = np.asarray(C.sum(axis=0)).ravel() ** 0.75
total = row.sum()
colsum = col.sum()
D = C.tocoo()
pmi = np.log((D.data * total * (colsum / total)) / (row[D.row] * col[D.col]))
pmi = np.maximum(0, pmi)
P = csr_matrix((pmi.astype(np.float32), (D.row, D.col)), shape=C.shape)

U, S, Vt = svds(P, k=DIM)
# weight by sqrt of singular values (standard trick), L2-normalize rows
E = U * np.sqrt(S)
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)

def vec(word):
    i = wid.get(word)
    return None if i is None else E[i]

# sanity probes
def near(word, k=8):
    v = vec(word)
    if v is None: return []
    sims = E @ v
    top = np.argsort(-sims)[:k + 1]
    return [(vocab[i], round(float(sims[i]), 3)) for i in top if vocab[i] != word][:k]

for probe in ["sea", "gull", "porter", "cloud", "green", "stone", "god"]:
    print(probe, "->", near(probe, 6), file=sys.stderr)

# the site paragraph + seed words
paragraph = sys.argv[2]
words = sorted(set(re.findall(r"[a-zà-ÿ]+(?:['’][a-zà-ÿ]+)*", paragraph.lower())))
seeds = ["sea", "water", "tide", "waves", "sky", "cloud", "clouds", "sun", "air",
         "rock", "stone", "cliff", "land", "green", "grass", "foam", "white", "surf",
         "froth", "spray", "wave", "shore", "grey", "dark", "light"]
need = sorted(set(words + seeds))
out = {}
missing = []
for w in need:
    v = vec(w)
    if v is None:
        # crude lemma fallback: strip plural/genitive s
        v = vec(w.rstrip("s")) if len(w) > 3 else None
    if v is None:
        missing.append(w)
    else:
        out[w] = [round(float(x), 4) for x in v]
print(f"exported {len(out)} vectors, missing: {missing}", file=sys.stderr)
open(sys.argv[3], "w").write(json.dumps(out, separators=(",", ":")))
print(f"wrote {sys.argv[3]}", file=sys.stderr)
