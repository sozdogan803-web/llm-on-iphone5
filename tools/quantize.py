"""llama2.c v0 float32 checkpoint -> int8 file for index.html.

Layout: 28-byte header (unchanged), rmsAtt, rmsFfn, rmsFinal (float32), then for each of
emb, wq, wk, wv, wo, w1, w2, w3: int8 values followed by one float32 scale per GS values.
Then int32 K, K int32 token ids (ascending, from keep.txt), and the same int8+scales form for those K
embedding rows: the pruned output layer, so the classifier only scores tokens the model really uses.
Usage: python quantize.py in.bin out.q8 keep.txt
"""
import sys
import numpy as np

GS = 32

src, dst = sys.argv[1], sys.argv[2]
keep = np.array(sorted({int(v) for v in open(sys.argv[3]).read().split()}), np.int32)
raw = open(src, "rb").read()
dim, hid, nl, nh, nkv, vocab, seq = np.frombuffer(raw, np.int32, 7)
assert vocab > 0, "expected a shared classifier (positive vocab_size)"
hs = dim // nh
kvd = dim * nkv // nh
pos = 28


def take(n):
    global pos
    a = np.frombuffer(raw, np.float32, int(n), pos)
    pos += int(n) * 4
    return a


emb = take(vocab * dim)
rms_att = take(nl * dim)
wq = take(nl * dim * dim)
wk = take(nl * dim * kvd)
wv = take(nl * dim * kvd)
wo = take(nl * dim * dim)
rms_ffn = take(nl * dim)
w1 = take(nl * hid * dim)
w2 = take(nl * dim * hid)
w3 = take(nl * hid * dim)
rms_final = take(dim)

out = [raw[:28], rms_att.tobytes(), rms_ffn.tobytes(), rms_final.tobytes()]
assert keep[0] >= 0 and keep[-1] < vocab and 1 in keep
tensors = [("emb", emb), ("wq", wq), ("wk", wk), ("wv", wv), ("wo", wo), ("w1", w1), ("w2", w2), ("w3", w3)]
for name, t in tensors + [("cls", None)]:
    if t is None:
        out.append(np.int32(len(keep)).tobytes() + keep.tobytes())
        t = emb.reshape(vocab, dim)[keep]
    assert t.size % GS == 0 and t.size % 4 == 0
    g = t.reshape(-1, GS)
    scale = np.abs(g).max(axis=1) / 127.0
    scale[scale == 0] = 1.0
    q = np.rint(g / scale[:, None]).astype(np.int8)
    err = np.abs(q * scale[:, None] - g).max()
    print("%-4s %9d values  max abs error %.6f" % (name, t.size, err))
    out += [q.tobytes(), scale.astype(np.float32).tobytes()]

blob = b"".join(out)
open(dst, "wb").write(blob)
print("wrote %s: %d bytes (was %d)" % (dst, len(blob), len(raw)))
