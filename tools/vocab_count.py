"""Samples stories from the float32 model (temperature 1, no top-p) and writes the token ids it ever uses.

Usage: python vocab_count.py model.bin keep.txt [batches] [coverage]
keep.txt gets one id per line: the fewest tokens covering `coverage` (default 0.99995) of all sampled
tokens, plus BOS/EOS. index.html only scores these in the output layer.
"""
import sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
batches = int(sys.argv[3]) if len(sys.argv) > 3 else 20
coverage = float(sys.argv[4]) if len(sys.argv) > 4 else 0.99995
B = 64

raw = open(src, "rb").read()
dim, hid, nl, nh, nkv, vocab, seq = [int(v) for v in np.frombuffer(raw, np.int32, 7)]
assert vocab > 0 and nkv == nh, "expects a shared classifier and no grouped-query attention"
hs = dim // nh
pos = 28


def take(*shape):
    global pos
    n = int(np.prod(shape))
    a = np.frombuffer(raw, np.float32, n, pos).reshape(shape)
    pos += n * 4
    return a


emb = take(vocab, dim)
rms_att = take(nl, dim)
wq, wk, wv, wo = take(nl, dim, dim), take(nl, dim, dim), take(nl, dim, dim), take(nl, dim, dim)
rms_ffn = take(nl, dim)
w1, w2, w3 = take(nl, hid, dim), take(nl, dim, hid), take(nl, hid, dim)
rms_final = take(dim)
freq = 1.0 / 10000 ** (np.arange(0, hs, 2) / hs)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * w


def rope(v, p):
    v = v.reshape(v.shape[0], nh, hs // 2, 2)
    c, s = np.cos(p * freq), np.sin(p * freq)
    return np.stack([v[..., 0] * c - v[..., 1] * s, v[..., 0] * s + v[..., 1] * c], -1).reshape(v.shape[0], dim)


rng = np.random.default_rng(1)
counts = np.zeros(vocab, np.int64)
first = []  # one sampled story, for eyeballing that the numpy forward pass is right
for b in range(batches):
    kc = np.zeros((nl, B, seq, nh, hs), np.float32)
    vc = np.zeros_like(kc)
    tok = np.ones(B, np.int64)  # BOS
    alive = np.ones(B, bool)
    for p in range(seq - 1):
        x = emb[tok]
        for l in range(nl):
            xb = rms(x, rms_att[l])
            q, k, v = rope(xb @ wq[l].T, p), rope(xb @ wk[l].T, p), xb @ wv[l].T
            kc[l, :, p], vc[l, :, p] = k.reshape(B, nh, hs), v.reshape(B, nh, hs)
            sc = np.einsum("bhd,bthd->bht", q.reshape(B, nh, hs), kc[l, :, : p + 1]) / np.sqrt(hs)
            sc = np.exp(sc - sc.max(-1, keepdims=True))
            sc /= sc.sum(-1, keepdims=True)
            x = x + np.einsum("bht,bthd->bhd", sc, vc[l, :, : p + 1]).reshape(B, dim) @ wo[l].T
            xb = rms(x, rms_ffn[l])
            u = xb @ w1[l].T
            x = x + (u / (1 + np.exp(-u)) * (xb @ w3[l].T)) @ w2[l].T
        logits = rms(x, rms_final) @ emb.T
        tok = np.argmax(logits + rng.gumbel(size=logits.shape), -1)  # exact sample at temperature 1
        counts += np.bincount(tok[alive], minlength=vocab)
        if b == 0 and alive[0]:
            first.append(int(tok[0]))
        alive &= tok != 1
        if not alive.any():
            break
    print("batch %d/%d, %d tokens so far" % (b + 1, batches, counts.sum()), flush=True)

order = np.argsort(-counts)
cum = np.cumsum(counts[order]) / counts.sum()
k = int(np.searchsorted(cum, coverage)) + 1
keep = np.unique(np.concatenate([order[:k], [1]]))
print("distinct tokens seen: %d; keeping %d (coverage %.5f)" % ((counts > 0).sum(), len(keep), cum[k - 1]))
for n in (2000, 4000, 8000, 16000):
    print("  top %5d cover %.5f" % (n, cum[n - 1]))
open(dst + ".sample", "w").write(" ".join(map(str, first)))
open(dst, "w").write("\n".join(str(int(i)) for i in keep))
