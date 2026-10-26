# A 15M-parameter LLM in the Safari of an iPhone 5

[stories15M](https://huggingface.co/karpathy/tinyllamas) (llama2.c architecture) running inference **on the phone**, in Mobile Safari on iOS 10.3.4, as plain ES5 JavaScript. No WebAssembly, no native app, no server doing the inference. The page is just served from a PC once; after that the phone only needs the page and an 18 MB model file.

**Measured on an iPhone 5 (A6, 2 cores, 1 GB RAM): 11.7 tokens/s**, up from 2.1 for the first working port. How it got there, and why each step worked or didn't: [docs/how-it-works.md](docs/how-it-works.md).

What got it there, briefly: two threads (Web Worker on a SharedArrayBuffer), int8 weights, and scoring only the 3,926 of 32,000 output tokens the model actually uses.

## Status: what is and isn't verified

| Claim | State |
|---|---|
| 11.7 tok/s on the phone | measured by me, on a build **before** the packed-load kernel |
| Packed-load kernel (`#pk`) | exact (bit-identical logits to `#int`) and ~24% faster than float on a PC, but **no gain on the iPhone 5**: 11.4 tok/s against a default-page peak of about 12 (within run-to-run noise). Use the default page. |
| Works offline (Application Cache) | implemented; **not tested on iOS 10** |
| Quality after int8 + output-layer pruning | read outputs, they are coherent stories; **no perplexity measurement** |

## Run it

Needs Python 3. The phone must be on the same network as the PC for the first load.

```bash
python app/serve.py
```

It prints your LAN addresses. Open `http://<that address>:8000` on the phone. Pages and flags:

| URL | Kernel |
|---|---|
| `/` | float activations (default) |
| `/#int` | int8 activations |
| `/#pk` | int8 activations, 4 weights per load (fastest on a PC, no gain on the iPhone 5) |
| `/#single` | one thread only (combine with the others, e.g. `/#single&pk`) |
| `/f32.html` | float32 baseline (needs `models/stories15M.bin`, see below) |

Open the "Model Parameters" drawer to see tok/s and a per-token timing split. `app/app.appcache` makes the page and model available offline: bump its `# vN` comment whenever `index.html` or the model changes, or the phone keeps the old copy.

## Rebuild the model

```bash
# stories15M.bin = karpathy/tinyllamas on Hugging Face (llama2.c v0 float32 checkpoint)
pip install numpy
python tools/vocab_count.py models/stories15M.bin models/keep.txt 16   # which output tokens the model uses
python tools/quantize.py models/stories15M.bin models/stories15M.q8 models/keep.txt
```

The 60 MB float32 checkpoint is not in this repo. `models/stories15M.q8` (18 MB) is.

## Credits and licence

- Model architecture, `tokenizer.bin` and the stories15M weights: Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c) and [tinyllamas](https://huggingface.co/karpathy/tinyllamas), both MIT. `stories15M.q8` is a derived (quantised, output-pruned) copy of those weights, trained on TinyStories.
- Code in this repo: MIT, see [LICENSE](LICENSE).

## Prior art

Not searched. The ingredients (llama2.c, int8 group quantisation, JS ports, vocabulary pruning) are all known; I have not checked whether this exact combination has been done on this hardware, so don't read "first" into it.
