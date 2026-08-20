"""Run TextFooler, rewrite each adversarial example with LLaMA-2, then filter.

Original pipeline from the project, with comments translated to English and
the semantic filter (see `semantic_filter.py`) wired in at step 6 — in the
original run the filter lived in a separate script that has since been lost,
so this is the reconstructed end-to-end path.

Usage:
    HF_TOKEN=... python src/run_attack.py
"""

import os

import textattack
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from refiner import AdversarialRefiner

VICTIM_MODEL = "textattack/bert-base-uncased-ag-news"
DATASET = "ag_news"
NUM_EXAMPLES = 100
POOL_SIZE = 500


def run_pipeline(output_csv: str = "results/textfooler_results22.csv"):
    # 1. Load the victim classifier (BERT fine-tuned on AG News).
    model = AutoModelForSequenceClassification.from_pretrained(VICTIM_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(VICTIM_MODEL)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # 2. Load the dataset and take a fixed slice of the test split.
    dataset = HuggingFaceDataset(DATASET, split="test")
    dataset._dataset = dataset._dataset.select(range(POOL_SIZE))

    # 3. Build the word-substitution attack used as the baseline.
    attack = TextFoolerJin2019.build(model_wrapper)

    # 4. Attack settings.
    attack_args = textattack.AttackArgs(
        num_examples=NUM_EXAMPLES,
        log_to_csv=output_csv,
        disable_stdout=True,
    )

    # 5. Run the attack.
    attacker = textattack.Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()

    # 6. Rewrite each adversarial example with LLaMA-2.
    refiner = AdversarialRefiner(device=0)
    for result in results:
        refined = refiner.refine_text(
            original=result.original_text(),
            perturbed=result.perturbed_text(),
            target_label=result.original_result.ground_truth_output,
        )
        result.perturbed_text = refined

    # 7. Save the refined results.
    #    To reproduce the filtered condition reported in the paper, pass the
    #    rewrites through semantic_filter.filter_candidates before logging.
    logger = textattack.loggers.CSVLogger("results/refined_results22.csv")
    logger.log_summary(results)


if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("Set HF_TOKEN; Llama-2 is a gated model on the Hub.")
    run_pipeline()
