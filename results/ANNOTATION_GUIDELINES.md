# Annotation guidelines: is this rewrite a valid adversarial probe?

You will see an original text and a rewritten version of it. You will **not** see
what any model predicted for either one, and you should not try to guess. Your
job is to judge the texts, not the models.

Answer three questions per item. Take about 45 seconds each; if an item takes
more than two minutes, mark `unsure` and move on.

---

## Q1. `label_preserved` — does the rewrite still have the same correct answer?

The correct answer is the one a careful human reader would give, not the one a
model gives.

Per dataset, the question is:

| Dataset | The label is | So ask yourself |
|---|---|---|
| IMDb, SST-2 | positive / negative sentiment | Does the rewrite still express the same verdict about the same thing? |
| AG News | world / sports / business / sci-tech | Is the rewrite still about the same topic? |

Answer:

- **`y`** — a reader would give both texts the same label.
- **`n`** — a reader would give them different labels. The rewrite changed the
  answer, so a model changing its prediction is behaving correctly.
- **`unsure`** — genuinely ambiguous, or the text is too damaged to judge.

**Worked examples.**

> *Original:* Sloppy film noir thriller which doesn't make much of its tension
> promising set-up. (3/10)
> *Rewrite:* Engaging film noir thriller that successfully builds on its
> intriguing premise. (8/10)

→ **`n`**. The verdict is reversed. This is not an attack, it is a different
review. (Note this item has cosine similarity 0.964 — do not let surface
similarity sway you.)

> *Original:* Yukos warns it oil output is lagging. Beleaguered Russian energy
> giant Yukos…
> *Rewrite:* Lakers facing setbacks in NBA title defense, LeBron's injury a major
> concern…

→ **`n`**. Business became sports. Different topic, different correct label.

> *Original:* while the resident evil games may have set new standards for
> thrills, suspense, and gore, the movie really only succeeds in the third of
> these.
> *Rewrite:* while the resident evil games may have set new standards for
> thrills, suspense, and gore, the movie surprisingly thrives in only one of
> these.

→ **`y`** or **`unsure`**. Both say the film delivers on only one of three
counts. The wording moved but the verdict did not. This is what a valid
adversarial probe looks like.

**The test to apply:** if you showed both texts to a friend and asked for the
label, would they answer the same way? If yes, `y`.

**Do not** mark `n` merely because the wording changed a lot. Heavy rewording
with the same verdict is exactly what we are looking for. Mark `n` only when the
*answer* changed.

---

## Q2. `well_formed` — is the rewrite a usable input for this task?

- **`y`** — it is a piece of text of the same kind as the original.
- **`n`** — it is not. Examples: it contains meta-commentary about the rewriting
  task ("In this revised version, I have…"); it is an instruction rather than a
  text; for a sentence-pair task the two segments have been merged into one.

Judge the format only. A well-formed text can still have a changed label; those
are separate questions.

---

## Q3. `is_refusal` — did the generating model decline the request?

- **`y`** — the text is the model refusing, e.g. "I cannot provide a revised
  text that could be used to mislead…", "As an AI language model, I…".
- **`n`** — anything else.

A refusal is also `well_formed = n`, but mark both.

---

## Notes column

Use it whenever you hesitate. Reasons you were torn are more useful to us than a
confident label. Especially note: sarcasm, mixed sentiment, ratings that
contradict the prose, or topics that plausibly fall under two categories.

---

## Rules

1. **Do not open `human_annotation_key.csv` before you finish.** It holds what
   the automatic signals said, and seeing it would contaminate the gold set that
   those signals are being scored against.
2. **Do not look up the model's prediction.** The point is to establish what is
   true, independent of what any model thinks.
3. Annotate in the order given; the order is already shuffled.
4. If you change your mind about an earlier item, go back and change it. Just do
   not revise after seeing the key.

---

## Second annotator

`human_annotation_overlap.csv` holds the first 50 items again. A second person
should annotate it without seeing your labels, so we can report Cohen's kappa.
Agreement is the evidence that the gold set means anything; a single annotator
grading their own experiment is not a gold standard.
