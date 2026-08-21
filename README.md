# When the Attack Is Not an Attack

Validity failures in LLM-generated red-teaming.
[Paper](paper/revised/main.pdf) · [One-page summary](paper/revised/summary.pdf)

Automated red-teaming increasingly uses one language model to generate the probes
that test another. The appeal is scale and fluency: an LLM can rewrite a stilted
word-substitution attack into text a person might actually write.

But a generated probe only tells you something about the target model if it is a
valid test of that model. This repository measures how often it is, and what the
answer does to the numbers such pipelines report. Four failures, each invisible
in the attack success rate that red-teaming reports usually carry:

| Failure | What goes wrong | Effect on the reported number |
|---|---|---|
| **Label validity** | the rewrite changed the input's true label, so a prediction change is correct behaviour | a referee model reproduces **86–87%** of the best-scoring prompt's successes; requiring target-specific flips **reverses** which prompt is better |
| **Input validity** | the rewrite is not a well-formed input for the task | **understates** — 12.5% against 41.7% on sentence-pair tasks |
| **Generator validity** | the generator refused, and its refusal is scored as a probe | **6.2%** of generations, at 11.0% vs. 0.6% depending on the prompt under study |
| **Measurement validity** | the reported metric is not a function of the generated text | undetectable from the outputs alone |

The first two push in opposite directions, so they do not cancel; which one
dominates depends on the task.

## The results

**Label validity.** A task-specific rewriting prompt roughly doubles the attacks
that survive rewriting — 119/180 against 58/180, dataset and target held fixed,
*p* = 1.7×10⁻¹⁰. On sentiment this is free (similarity 0.713 vs. 0.705). On topic
classification similarity collapses from 0.691 to **0.361**, because that prompt
instructs the model to change the topic "from world to sports". A text whose topic
really changed is a different input with a different correct label.

> The instruction most effective at producing label flips is the instruction to
> change the label. Any prompt search optimizing reported attack success finds it.

**The similarity check is not enough.** It catches the AG News case and misses
sentiment, because sentence embeddings track topic rather than verdict. This IMDb
rewrite has cosine similarity **0.964**:

> *Sloppy film noir thriller which doesn't make much of its tension promising
> set-up.* **(3/10)** → *Engaging film noir thriller that successfully builds on
> its intriguing premise.* **(8/10)**

**A referee model separates attacks from label changes.** An adversarial example
should fool *one* model, not persuade every model, so each probe is re-scored with
an independently trained classifier of the other architecture. 86–87% of the
task-specific prompt's successes flip the referee too — and requiring the flip to
be target-specific reverses the ranking:

| Prompt | Flip rate | Genuine vulnerabilities |
|---|---|---|
| task-specific | **119/180** (0.661) | 16/180 (0.089) |
| generic | 58/180 (0.322) | **29/180** (0.161) |

Of the 119 flips, 39 survive the similarity threshold, 16 survive the referee, and
**only 5 survive both** — if the two checks caught the same thing the overlap would
be near 16. They fail in different places: the similarity check waves through the
3/10-to-8/10 review, while the referee misses a world-news article rewritten into a
rugby story that similarity caught at 0.268. Adding one check is not enough.

Counting a probe as target-specific whenever the referee does not flip is also
generous — the referee may have disagreed with the target on the original to begin
with. The two agree on the original 94.4% of the time; requiring agreement leaves
**12/180** genuine vulnerabilities for the task-specific prompt against **25/180**
for the generic one, so the reversal widens rather than softens.

| Dataset | Flip rate | Meaning-preserving | Ratio |
|---|---|---|---|
| AG News | 65.0% | **3.3%** | 19.5× |
| SST-2 | 50.0% | 15.0% | 3.3× |
| IMDb | 83.3% | 46.7% | 1.8× |
| **Pooled** | **66.1%** | **21.7%** | **3.05×** |

**Input validity.** Sentence-pair tasks need two segments; most probes merge them
(100 of 198 well-formed across the stored corpus, and 0 of 31 for RTE under PWWS).
Malformed probes never register a flip, so leaving them in the denominator counts
them as attack failures. An explicit format constraint nearly doubles validity
(30.0% → 57.5%, *p* = 2.8×10⁻⁵) but the model still violates a stated format more
than 40% of the time.

**Generator validity.** The generator is itself safety-trained and sometimes
declines: *"I cannot provide a revised text that could potentially be used to
mislead or deceive..."* Of 600 generations, **37 (6.2%)** begin with a refusal and
**7 of those are scored as successful attacks** (mean similarity 0.216). It is not
evenly spread over the conditions being compared — 11.0% under the generic prompt
against 0.6% under the task-specific one, and 17.5% on RTE — so it is a confound in
the comparison above, not just noise. Excluding refusals leaves both results intact
(118/179 vs. 52/170, *p* = 4.2×10⁻¹¹), but a refusal and a failed attack look
identical in the output column.

**Measurement validity.** Section 8 of the paper works through an instance: an
earlier version of this study wrote LLaMA output into a `llama_text` column while
computing the reported success rate from TextAttack's `result_type` field, which
describes the word-substitution attack that ran *before* rewriting. The check that
catches it is one line — assert that the evaluated column is the generated column.

## Underneath: rewriting is a trade

Rewriting is applied to attacks that already succeed, so its cost is directly
measurable. Across 32 configurations, only **44.5%** of those attacks survive.
What it buys is fluency: median GPT-2-XL perplexity **22.9** against **522** for
the TextFooler perturbations being replaced.

The generate-and-filter pipeline proposed for this setting retains 15.3% of
generations, and the **similarity** constraint does nearly all of the filtering —
it removes 113 of the 177 probes that flip, while the fluency bound removes 9 more.

| Stage | Kept | Fraction | Mean sim. / median ppl |
|---|---|---|---|
| all probes | 360 | 1.000 | 0.648 / 33.8 |
| + flips the classifier | 177 | 0.492 | 0.596 / 28.1 |
| + similarity ≥ 0.75 | 64 | 0.178 | 0.840 / 27.2 |
| + perplexity ≤ 150 | 55 | 0.153 | 0.834 / 22.9 |

## Five numbers a red-teaming report should carry

1. **Semantic similarity beside every flip rate.** Without it, a prompt that
   changes the label scores best.
2. **The fraction of generations that are well-formed inputs.** Without it,
   malformed probes sit in the denominator as failures.
3. **The fraction of successes an independent referee reproduces.** Similarity
   alone misses 56% of the label changes on IMDb.
4. **The generator's refusal rate, per condition.** Without it, a generator that
   declines more often under one prompt than another moves the comparison.
5. **The number of generations actually scored by the target model.** Without it,
   a pipeline can report a metric computed from something else, and nothing in the
   output will look wrong.

## Layout

```
src/semantic_filter.py               per-candidate three-stage filter
src/refiner.py                       LLaMA-2 rewriting
src/run_attack.py                    TextFooler → rewrite → log
src/custom_attack.py                 TextFooler with constraints removed
scripts/prompt_study.py              the prompt and structure ablations
scripts/reevaluate_llama_rewrites.py runs the target classifier on stored probes
scripts/run_filter_pilot.py          filter applied to recorded attack results
scripts/referee_check.py             re-scores probes with the other architecture
scripts/make_tables.py               builds the result tables
results/                             every table in the paper, plus raw generations
paper/revised/                       paper and one-page summary (LaTeX + PDF)
paper/                               the 2024 version, for reference
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...          # Llama-2 is gated on the Hugging Face Hub
export NLTK_DATA=...         # only if WordNet is not in the default location
```

A single 24 GB GPU is enough. The scripts load the generator, free it, and only
then load the scorers, so peak memory stays near the size of Llama-2 alone
(13.5 GiB observed).

## Running

```bash
# re-evaluate stored probes against their target classifiers
python scripts/reevaluate_llama_rewrites.py "PATH/TO/*_llama.csv" --device cuda:0

# the prompt and structure ablations (600 generations)
python scripts/prompt_study.py --configs "$(cat configs.json)" --n 30 --device cuda:0 \
    --out results/prompt_study.csv

# rebuild every table in the paper
python scripts/make_tables.py results/prompt_study.csv --outdir results
```

## Provenance

`src/semantic_filter.py` is a reference implementation written from the algorithm
in the 2024 paper; the earlier pipeline applied its similarity constraint at the
attack layer and computed quality statistics per file rather than per candidate.
Everything under `scripts/` and every table in `results/` is from runs made for
the revised study. The stored 2024 probes are re-evaluated, not regenerated.

## Attribution

Built on [TextAttack](https://github.com/QData/TextAttack) (QData, MIT licensed),
which provides the attack recipes, model wrappers, and dataset loaders. TextAttack
is a dependency, not a copy. TextFooler is due to Jin et al. (2020), PWWS to Ren
et al. (2019), and LLaMA-2 to Touvron et al. (2023).

Work done as a research assistant at FOCAL Lab, UIUC, with Prof.
[Gagandeep Singh](https://ggndpsngh.github.io/), and revised independently in 2026.

## Contact

Yihang Jiao — yihangj3@illinois.edu · https://yhj3.github.io
