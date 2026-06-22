---
layout: project
title: "Can a Transformer Learn Position in a Few Thousand Parameters?"
summary: "Replacing nanoGPT's 131,072 position-embedding parameters with a 3,456-parameter periodic attention bias on enwik8, a 38× reduction with no measurable loss in performance."
thumbnail: /assets/img/pd_curves.png
repo: https://github.com/JamesSteiner/nanoGPT
date: 2026-06-20
---

Transformers don't natively know the order of their inputs. Self-attention treats a
sequence as a set, i.e. "the cat sat on the mat" and "the mat sat on the cat" look
identical to it, so every transformer needs an extra signal telling it where each
token sits. The usual fix is a positional embedding. The most common version is a learned lookup
table: one trainable vector per position, added to the token embeddings at the input,
as used by GPT-2 and many models since. It works, but it's a slightly unsatisfying fix. The table can't describe any position past the length it was
trained on, and it makes position 5 and position 105 each learn from scratch that
their neighbours behave the same way.

I wanted to know how *little* a model actually needs in order to encode position. My
testbed is nanoGPT, Andrej Karpathy's minimal GPT-2 reimplementation, trained
character-level on enwik8 (UTF-8 encoded text extracted from Wikipedia containing markup and XML). I took its learned position-embedding table (131,072
parameters in this setup, about 0.3% of the model) and replaced it with something far
smaller: a per-head bias added inside attention, written as a short sum of cosines of
the distance between two tokens. The hope was that a Fourier
basis would let each head *discover* whatever periodic structure is useful in the
data, like markup tags, line lengths, indentation in Wikipedia markup, rather than
memorising absolute slots.

And it works: on enwik8, a model with no position embeddings at all leaning
entirely on ~3,500 cosine parameters (0.008% of the network) matches the baseline.
The more interesting question is what those cosines became. I half-expected the
Fourier basis to collapse into a plain recency penalty: the "attend to nearby tokens"
decay that **ALiBi** hard-codes by hand. There is a clear recency tilt but it
isn't the whole story. Every single head also learned a genuinely oscillatory,
multi-scale bias, with roughly half of its variation periodic rather than monotonic;
the model took the Fourier basis up on its offer. What I can't yet say is whether
the periodicity actually earns its keep, a plain recency decay might do as well and
I haven't tested length extrapolation, the one thing this whole family of methods is
meant to be good at. This post is about that result: what can replace the embeddings,
and what the model learns when you let it.

## How transformers represent position

Because attention is permutation-invariant, every approach to position is really a
choice about what extra information to inject, and where. The original transformer
injected it at the input as fixed sinusoidal vectors (Vaswani et al., 2017); GPT-2,
and so nanoGPT, swapped those for the learned absolute table from the intro. Either
way the signal is absolute, with the limitation we've already met: it can't describe a
position past the training length. That is what the next family sets out to fix.

A second family moves position *inside* the attention computation and makes it
relative, depending on the offset `i − j` between two tokens rather than their
absolute indices. Shaw et al. (2018) learned a separate vector for each offset; T5
(Raffel et al., 2020) pared this down to a learned scalar bias per bucketed distance.
ALiBi (Press et al., 2022) went further and removed the learning entirely, adding a
fixed linear penalty proportional to distance that nudges every head toward recent
tokens and, as a side effect, extrapolates remarkably well to longer sequences.
KERPLE (Chi et al., 2022) generalised this into a family of learnable distance
kernels with ALiBi as one special case, and its sibling Sandwich expressed the bias
as a sum of cosines of the offset; FIRE (Li et al., 2023) swapped the kernel for a
small learned function of normalised distance. RoPE (Su et al., 2021) takes a
different route again, rotating queries and keys so that their dot product depends on
relative position multiplicatively rather than through an additive bias and this is now
the default in most open LLMs.

A final and more surprising option is to add nothing at all. Because the causal mask
already breaks the symmetry between positions, a decoder-only transformer can recover
positional information with no encoding whatsoever (Haviv et al., 2022), and such
"NoPE" models can even out-generalise explicit schemes on small algorithmic tasks
(Kazemnejad et al., 2023). At scale, though, NoPE tends to fall behind: DroPE
(Sakana, 2025) argues both empirically and theoretically that NoPE models train less
efficiently than RoPE ones, and proposes instead to keep RoPE during pretraining and
*remove* it afterward, treating position encoding as a kind of training scaffold.

That last strand matters for reading my results. A causal model already recovers some
sense of position for free, so "delete the embeddings and it still works" is partly
expected; the real question is whether a few thousand learned parameters buy anything
beyond that, and what shape they take when they do. To be clear, my periodic-bias
model is not a NoPE model, it still injects an explicit relative signal into
attention. True NoPE, with no embeddings and no bias, is a baseline I return to
later.

## The mechanism

Why cosines? enwik8 is raw Wikipedia markup, and markup is dense with quasi-periodic
structure: XML tags that open and close at fairly regular spacing, indentation, list
markers, repeated template boilerplate. A sum of sinusoids is the natural basis for
patterns like these since low frequencies can capture broad positional trends and high
frequencies can capture short repeats, so rather than fix the frequencies in advance, I let each
attention head learn the handful that matter to it. (As the results section will show,
the model took this up more than I expected, every head ends up genuinely oscillatory
though whether the periodicity actually helps is a separate question.)

For each pair of positions `(i, j)`, with relative distance `d = i − j`, I add to
the attention logits (before softmax) a per-head bias:

```
P(d) = Σ_{k=1}^{K} a_k · cos(ω_k · d + φ_k)
```

where amplitude `a_k`, angular frequency `ω_k`, and phase `φ_k` are **learned per
head**. With 8 heads, K=12 modes, 3 params/mode, that's 288 params/layer = **3,456
params total** across 12 layers versus 256×512 = 131,072 params in the learned
absolute embedding table.

This is a *translation-invariant (Toeplitz) relative-position bias*,
parametrized in a Fourier basis, a close relative of Sandwich, KERPLE, and TISA. The
only real twist is that the frequencies `ω_k` are learned rather than fixed. (As
we'll see, this twist is real but its benefit is untested.)

I compare three configurations: a **baseline** with learned absolute position
embeddings only (stock nanoGPT); **PE + periodic bias**, which keeps the embeddings
and adds the cosine bias; and **periodic bias only**, which removes the embeddings
entirely and lets the bias carry all positional information.

One implementation note: because the bias is added as an explicit `T×T` matrix before
softmax, this path does not use FlashAttention, so the bias variants are slower
per step than the baseline. At 256 context that's a minor cost; at long context it
would matter.

## Results

enwik8, treated as character-level over its UTF-8 characters (a ≈5,800-symbol vocab),
split 90M/5M/5M by character. All three models are identical apart from the positional
mechanism: 12 layers, 8 heads, d=512, block_size=256, ≈41M parameters, 75k iters,
AdamW at lr=2.5e-4, a single seed (1337).

| Configuration        | Train BPC | Val BPC |
|----------------------|:---------:|:-------:|
| Baseline (abs PE)    |   1.222   |  1.323  |
| PE + Periodic Bias   |   1.184   |  1.312  |
| Periodic Bias Only   |   1.204   |  1.312  |

(I report validation only. The repo's test split was later regenerated with a
different vocabulary than the trained checkpoints use, so I can't currently reproduce a
test number I'd stand behind; regenerating a consistent test set is on the to-do list.)

![Validation BPC over training](/assets/img/validation-curve.png)

Two readings, stated carefully. The first is that **the replacement works**: deleting
the position-embedding table and relying on 3,456 cosine parameters lands right at the
baseline (1.31 val, versus 1.32). This is the result I'm most confident in, and it's
visible throughout training with the no-PE curve tracking the baseline the whole way. The
second is that **the "improvement" is within noise**: adding the bias on top of the
embeddings buys about 0.01 BPC, but with a single seed and step-to-step wiggle larger
than that gap, I would not claim it as a reliable gain since it could flip with another
seed.

A caveat on the metric: I report cross-entropy in bits per character over the
≈5,800-symbol UTF-8 vocabulary, which is not the standard byte-level enwik8 task
(256 symbols, scored in bits per byte). So these numbers aren't comparable to the
enwik8 leaderboard. For orientation only, byte-level SOTA is around 1.0 bpb, my results are notably higher than these due to training over fewer epochs. What is
fair is the internal comparison: all three models share an identical setup and
budget, so the differences between them are the point, not the absolute values.

## What did the bias actually learn?

This is the part I find most interesting. The cleanest way to see it is to plot
the bias each head adds to its logits as a function of how far back the key is, that is
`P(d)` directly, straight from the trained no-PE checkpoint.

![Per-head learned bias P(d), by layer](/assets/img/pd_curves.png)

*Each panel is one layer; each curve is one of its 8 heads. x-axis: distance into
the past (i − j); y-axis: bias added to the logit before softmax.*

Two things stand out, and they coexist. There is a clear *recency tilt*: 97% of
heads place more weight on nearby keys than distant ones, which is the same instinct ALiBi
hard-codes by hand, here discovered from a Fourier basis with no recency prior built
in. But the curves are also unmistakably oscillatory. Every one of the 96 heads
has interior wiggles (a median of 12 of them), and by a simple energy measure about
half of each head's bias variation comes from modes that complete a full period or
more across the window. The oscillation is strongest in the early and middle layers
and gentler, but never absent, in the deepest ones; not a single head collapses to a
plain monotonic decay. So the model didn't merely re-derive a recency penalty: it
layered genuine, multi-scale periodicity on top of one. Whether that periodicity earns
its keep, versus a plain monotonic recency bias, is the ablation I
come back to at the end.

To decompose those curves into ingredients, I pulled all 1,152 learned frequencies
(12 layers × 8 heads × 12 modes) out of the same checkpoint.

![Learned frequencies: folded \|ω\| and wavelengths vs. the window](/assets/img/frequency_analysis.png)

*Left: frequencies folded to |ω| (signed ω is degenerate, see below). The dashed
line marks one full period per 256-char window; 69% of modes (by count) lie to its right.
Right: the same modes as wavelengths (λ = 2π/|ω|, characters, amplitude-weighted),
spanning ~20-char ripples to window-spanning trends, median ≈ 90 chars.*

A histogram of signed `ω` (which is how it's tempting to plot it first) is
centered at zero and looks like the model gave up on periodicity. It didn't, for
three reasons:

1. **The sign of ω is a degeneracy.** `cos(ω·d + φ)` with `−ω` equals
   `cos(ω·d − φ)`: flipping the sign of ω is absorbed by the phase. So symmetry
   around zero is an artifact of plotting *signed* ω; only `|ω|` is meaningful.
2. **What matters is wavelength vs. window**, `λ = 2π/|ω|` against the 256-char
   context, not ω in the abstract. Amplitude-weighted: median `|ω| ≈ 0.07`, i.e.
   **λ ≈ 90 characters (~2.8 oscillations across the window)**. The percentiles span
   λ ≈ 317 / 90 / 33 / 22 characters (p25/p50/p75/p90), a multi-scale
   spread from window-spanning trends down to ~20-char ripples. ~67% of modes
   (amplitude-weighted) complete at least one full period inside the window; only
   ~3–7% sit in the near-constant regime.
3. **Higher-frequency modes carry more amplitude, not less:**
   `corr(|ω|, |amp|) = +0.41`. If the model were collapsing toward a flat bias,
   you'd see the opposite.

The no-PE and PE+bias checkpoints give near-identical statistics, so this is
structure, not noise.

One important caveat to mention is that this shows the model *chooses* to use multi-scale
periods and it does not *prove* periodicity actually helps. A smooth, monotonic distance
decay (à la ALiBi) might reach the same loss. The clean test is an ablation:
replace the learned oscillatory modes with a monotonic bias and see if loss rises.
I haven't run it. That's the first thing I'd do next.

## Limitations

A few things bound what this experiment establishes. The mechanism is not new:
learned additive relative biases over `i − j`, cosine parametrizations included,
already appear as TISA, Sandwich, and KERPLE. What's particular here is letting the
frequencies be learned rather than fixed, and I haven't yet shown that this freedom
buys anything over a fixed schedule however it's interesting to see what it learned
without baked-in priors. The study is also deliberately small with just a single
seed, one dataset (enwik8), one scale (≈41M) so the ~0.01 BPC edge from adding the
bias sits within seed noise; the result I'd stand behind is the match, not an
improvement.

The main gap is length extrapolation. Generalising past the training length is a big 
reason additive relative biases exist in the first place (ALiBi, KERPLE, FIRE), and I
trained and evaluated at 256 throughout so the axis where this approach should have
an edge is exactly the one I haven't measured. The comparison set is narrow, too: I
tested against learned absolute embeddings, where the more informative baselines are
RoPE, ALiBi, T5 bias, and true NoPE (the last would show whether the bias beats simply
letting the causal mask carry position). And the explicit-matrix bias gives up
FlashAttention, which would matter at long context.

## Future work

A handful of experiments would turn this from a demonstration into a claim, in rough
priority order. The first is **length extrapolation**: train at 256, evaluate at
512/1024/2048, and see whether the bias (like ALiBi) holds up while learned absolute
embeddings fall apart. The second is **the missing baselines** in the same harness:
RoPE, ALiBi, T5 bias, and NoPE. The third is **the periodicity ablation**: swap
the learned oscillatory modes for a monotonic bias and check whether the periods are
actually load-bearing. Fourth, **several seeds** with mean ± std, to firm up or retract
the small differences. And finally, **a second dataset or scale** like text8, or a larger
model for external validity.

## Takeaway

You can delete a transformer's learned position-embedding table and replace it
with ~3,500 cosine parameters in attention at no measurable cost on enwik8, and
the model learns interpretable multi-scale periodic biases when you do. The 
mechanism isn't new; it sits in the well-studied family of relative-position biases, and the decisive experiments, specifically
length extrapolation, the periodicity ablation, more seeds, stronger baselines, are
still ahead. What I find most interesting is that, handed a Fourier basis and no
instructions, the model reached for the same recency prior the field eventually
hard-coded as ALiBi, then built structure on top of it as a sort of validation for the hard-coded recency prior.

**Code:** https://github.com/JamesSteiner/nanoGPT a fork of
[Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). All of my changes
are a small diff on top: the `PeriodicBias` module in `new_model.py` /
`new_model_no_pe.py`, plus configs and training logs.

### References

- Vaswani et al. (2017), *Attention Is All You Need*.
- Shaw et al. (2018), *Self-Attention with Relative Position Representations*.
- Raffel et al. (2020), *T5*.
- Su et al. (2021), *RoFormer / RoPE*.
- Wennberg & Henter (2021), *The Case for Translation-Invariant Self-Attention* (TISA).
- Chi et al. (2022), *KERPLE*; Chi et al. (2022), *Dissecting / Sandwich*.
- Press et al. (2022), *ALiBi*.
- Haviv et al. (2022), *Transformer LMs without Positional Encodings Still Learn Positional Information*.
- Kazemnejad et al. (2023), *The Impact of Positional Encoding on Length Generalization* (NoPE).
- Li et al. (2023), *FIRE*.
- Sakana AI (2025), *DroPE: Extending the Context of Pretrained LLMs by Dropping their Positional Embeddings*, arXiv:2512.12167.
