# A language model in the Safari of an iPhone 5

A 15-million-parameter language model (stories15M, Karpathy's llama2.c architecture) running **entirely on a 2012 iPhone 5**, in Safari, as plain JavaScript. No app install, no WebAssembly, no server doing the inference.

| | |
|---|---|
| Device | iPhone 5 (iPhone5,2), A6 SoC, 2 cores, 1 GB RAM, 32-bit armv7s |
| OS / browser | iOS 10.3.4 (the last version this phone can run), Mobile Safari |
| Model | stories15M: dim 288, hidden 768, 6 layers, 6 heads, vocab 32000, context 256 |
| Result | **11.7 tokens/s** measured on the phone (started at 2.1) |

## Why this is hard

- **No WebAssembly.** iOS 10 Safari has no usable WASM, so every multiply-add runs as JavaScript. The code is ES5 only.
- **No fast math libraries, no GPU path.** No WebGL compute, no WebGPU, no Metal access from a web page.
- **Weak CPU.** The A6 does roughly 85 million multiply-adds per second in JS on this model, about 30 cycles each. It is bound by the JIT and by compute, not by memory bandwidth. This was measured: shrinking the weights from float32 to int8 alone gave no speedup (6.7 -> 6.6 tok/s).
- **Little memory.** 1 GB total, with Safari allowed only a fraction of it. The float32 model is 60 MB; the final file is 18.4 MB.
- **No service workers** (iOS 10 predates them), so offline use needs the old Application Cache.

## Speed history, as measured on the phone

| Step | tok/s |
|---|---|
| First working port (float32, one thread) | 2.1 |
| JS-level tuning (flat typed arrays, unrolled loops, independent accumulators) | 3.9 -> 4.2 |
| Second thread (Web Worker + SharedArrayBuffer, fused dispatch) | 6.7 |
| int8 weights, float activations | 6.6 (no gain; the phone is compute-bound) |
| **Vocabulary pruning** (output layer 32000 -> 3926 tokens) | **11.7** |
| Packed int32 loads (`#pk`) | **11.4: no gain** (default page peaks at about 12); ~24% faster on a PC |

The three steps that mattered, in order of payoff, are below.

## 1. Cut the work, not just the cost of the work: vocabulary pruning

The output layer scores every token in the vocabulary: 32000 x 288 multiply-adds per token, which is a large share of the whole forward pass for a 15M model. But stories15M was trained on TinyStories, a narrow corpus, so it only ever writes a small part of the vocabulary. `tools/vocab_count.py` samples stories from the float32 model itself (1024 stories at temperature 1, no top-p, about 200k tokens) and counts which token ids it picks. Only **3936 distinct tokens ever appear**; the top 4000 cover 100% of the sample. The final model keeps 3926 (99.995% of sampled tokens) and stores the id of each kept token, so the sampler works over those and maps back to real token ids. Input tokens are unaffected: the embedding table still holds all 32000 rows, so any prompt still works.

Consequence: the classifier shrinks about 8x, the file shrinks, and tokens/s jumps from 6.6 to 11.7. Trade-off: the model can no longer emit a token outside that set. The set comes from the model's own samples, so tokens it would pick with probability well below ~1 in 20,000 are dropped. That is nearly free for a story model, but it is lossy in principle and the coverage is an estimate from a finite sample.

## 2. Use both cores from a web page

The A6 has two cores. JS is single-threaded, so the model runs a second **Web Worker** on a **SharedArrayBuffer** that holds every vector both threads touch. Matmuls that read the same input are queued together (`mq()`) and released in one go (`go()`), so there is **one synchronisation per group of matmuls**, not one per matmul. Sync is a small Int32Array control block (`go`, `done`, job count, kind, head range, kernel mode). Attention is split by head across the two threads. The second thread's idle time is now measured and shown in the UI.

Many browsers dropped SharedArrayBuffer after Spectre and only expose it to cross-origin-isolated pages, so the server sends COOP/COEP headers for desktop testing. iOS 10 Safari still had it.

## 3. int8 quantisation, and then making the integer kernel less wasteful

`tools/quantize.py` converts the float32 checkpoint to int8 with one float32 scale per 32 weights (group size 32), max abs rounding error about one quantisation step. Activations are quantised the same way on the fly, so the inner loop becomes integer multiply-adds, with one float multiply per 32-wide group.

Plain int8 was *not* faster on the phone, because the cost is in the loop, not the bytes. The packed kernel (`mmP`) reads **four int8 weights per Int32Array load** and splits them with shifts (`(a<<24)>>24`, `(a<<16)>>24`, `(a<<8)>>24`, `a>>24`). That removes three quarters of the typed-array reads and bounds checks per multiply-add. It is bit-identical to the plain int kernel, and on a PC it was about 35% faster than int8 and 24% faster than float at the same thread count. On the iPhone 5 it gave no gain (11.4 tok/s against a default-page peak of about 12, within run-to-run noise), so the default page is still the one to use. My guess, untested, is that the A6's JIT pays about as much for the shifts as it saves on array reads.

## Smaller pieces

- **File format.** One `.q8` file, 18,389,200 bytes: a 28-byte header, float32 RMS norm weights, then int8 values plus scales for every tensor, then the list of kept token ids and the pruned output layer. Every tensor starts 4-byte aligned so Int32Array views are valid over the same buffer.
- **UI speed.** Waiting on `setTimeout` between tokens cost about 4 ms each. The UI now generates tokens back-to-back for 30 ms per tick, which took the PC from 76 to ~121 tok/s with no change to the kernel.
- **Offline.** `app.appcache` caches the page, the model and the tokenizer, so after one download the phone runs it with no network. The page shows an "Offline copy" status line. Flags (`#single`, `#int`, `#pk`) live in the URL hash so they do not create separate cache entries.
- **Instrumentation.** A "Last reply, ms per token" line splits time into matmul, thread-2 wait, attention and other, so the next bottleneck is measured, not guessed.

## What did not work

- **Double-packing two products into one float64 ("SWAR").** Looks attractive, but the middle-coefficient trick needs three ~19-bit fields, which does not fit a double's 53-bit mantissa. Not built.
- **int8 weights on their own** for speed. Useful for file size, useless for the phone's tokens/s.
- **Any inference outside the phone.** Not used; this project is about the phone running the model.

## Is it new?

Honest answer: I do not know, and I have not searched. Each ingredient is established: llama2.c, int8 group quantisation, pure-JS and WASM ports of it, SharedArrayBuffer workers, vocabulary pruning, Application Cache. What I cannot say is that anyone has shipped this combination on an iPhone 5 / iOS 10 / no-WASM target at double-digit tokens/s. The parts that feel least standard are (a) pruning the output layer from corpus statistics to cut classifier cost 8x, and (b) getting integer kernels to pay off on a JIT-bound 32-bit CPU by cutting typed-array loads rather than memory traffic. I would call it "a nice engineering result on very old hardware" and only call it a first after someone searches for prior art. I can run that search if you want.

## Still unverified

- Offline caching and storage quota behaviour on iOS 10.
- How Safari renders the starter buttons and the drawer.
- Quality loss from int8 and pruning has only been checked by reading outputs (coherent stories), not by a perplexity measurement.
