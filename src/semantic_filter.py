"""Three-stage semantic filter for generated adversarial candidates.

REFERENCE IMPLEMENTATION.
=========================
This file was written from Algorithm 1 of the accompanying paper
("Using Counterfactuals to Achieve TextAttack: Experiments with TextFooler
and LLaMA-2", Section 3.2). It is *not* the original script used to produce
the numbers reported in the paper — that script was lost. The results in
`results/` and in the paper come from the original experimental run, not
from this file. The thresholds and the filtering logic below follow the
paper's specification exactly; anything not specified there is marked with
a NOTE and a stated assumption.

The filter keeps a generated candidate only if all three conditions hold:

    flip rate   phi(g) >= 0.6      candidate changes the classifier's decision
    similarity  s(g)   >= 0.75     candidate still means what the original meant
    perplexity  p(g)   <= 150      candidate reads as fluent English

The point of the filter is that an LLM rewrite is only evidence of a model
weakness if it stays faithful to the original input. Without the similarity
and perplexity constraints, a "successful attack" may simply be a sentence
that means something different, which tells you nothing about robustness.
"""

from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FLIP_RATE_MIN = 0.6
SIMILARITY_MIN = 0.75
PERPLEXITY_MAX = 150.0


@dataclass
class Candidate:
    """One generated rewrite of an original input."""

    text: str
    flip_rate: float = 0.0
    similarity: float = 0.0
    perplexity: float = float("inf")

    def passes(self) -> bool:
        return (
            self.flip_rate >= FLIP_RATE_MIN
            and self.similarity >= SIMILARITY_MIN
            and self.perplexity <= PERPLEXITY_MAX
        )


class PerplexityScorer:
    """GPT-2-XL perplexity, as specified in the paper."""

    def __init__(self, model_name: str = "gpt2-xl", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str) -> float:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc.input_ids.to(self.device)
        if input_ids.numel() < 2:
            return float("inf")
        loss = self.model(input_ids, labels=input_ids).loss
        return float(torch.exp(loss))


def flip_rate(
    predict: Callable[[str], int],
    original_label: int,
    variants: Sequence[str],
) -> float:
    """Fraction of sampled variants whose predicted label differs from the original.

    NOTE. The paper writes phi(g) = I(f(x) != f(g)), an indicator, but then
    thresholds it at 0.6 — a cut that is only meaningful for a rate. The paper
    also states that several variants are drawn per input via nucleus sampling.
    We therefore read phi as the flip rate *across the sampled variants of one
    input*, which is the only reading under which the 0.6 threshold does work.
    """
    if not variants:
        return 0.0
    flips = sum(1 for v in variants if predict(v) != original_label)
    return flips / len(variants)


def cosine_similarity(embed: Callable[[str], "torch.Tensor"], a: str, b: str) -> float:
    """Cosine similarity between sentence embeddings of `a` and `b`.

    NOTE. Algorithm 1 in the paper prints s(g) = 1 - cos(E(x), E(g)) but then
    requires s(g) >= 0.75. Read literally that would keep only rewrites that
    drift *far* from the original, which contradicts the surrounding text
    ("semantic preservation ... how close the adversarial example's meaning is
    to the original"). We implement plain cosine similarity, which is what the
    threshold and the prose describe.
    """
    va, vb = embed(a), embed(b)
    return float(torch.nn.functional.cosine_similarity(va.flatten(), vb.flatten(), dim=0))


def filter_candidates(
    original: str,
    original_label: int,
    generated: Sequence[str],
    predict: Callable[[str], int],
    embed: Callable[[str], "torch.Tensor"],
    scorer: PerplexityScorer,
    variants_per_candidate: int = 1,
) -> List[Candidate]:
    """Apply the three-stage filter and return the candidates that survive."""
    kept: List[Candidate] = []
    for text in generated:
        variants = [text] * variants_per_candidate
        cand = Candidate(
            text=text,
            flip_rate=flip_rate(predict, original_label, variants),
            similarity=cosine_similarity(embed, original, text),
            perplexity=scorer.score(text),
        )
        if cand.passes():
            kept.append(cand)
    return kept


def filter_report(generated: Sequence[str], kept: Sequence[Candidate]) -> str:
    """Human-readable summary, mirroring what the paper reports about the filter."""
    total = len(generated)
    if total == 0:
        return "no candidates"
    discarded = total - len(kept)
    return (
        f"{total} candidates, {len(kept)} kept, {discarded} discarded "
        f"({discarded / total:.1%})"
    )
