"""TextFooler with its semantic constraints removed.

Used as an upper bound on attack success: it shows how much of TextFooler's
success rate comes from the search itself rather than from the constraints
that keep its substitutions meaning-preserving.

Set NLTK_DATA if your WordNet data is not in the default location.
"""

import os

import nltk

if os.environ.get("NLTK_DATA"):
    nltk.data.path = [os.environ["NLTK_DATA"]]

from textattack import Attacker
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from transformers import AutoModelForSequenceClassification, AutoTokenizer

VICTIM_MODEL = "textattack/bert-base-uncased-imdb"


class CustomTextFooler(TextFoolerJin2019):
    @staticmethod
    def build_constraints():
        return []


def main():
    # Load the victim model and tokenizer.
    model = AutoModelForSequenceClassification.from_pretrained(VICTIM_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(VICTIM_MODEL)
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

    # Load the dataset.
    dataset = HuggingFaceDataset("imdb", split="test")

    # Build the attack.
    attack = CustomTextFooler.build(model_wrapper)
    attacker = Attacker(attack, dataset)

    # Run the attack.
    attacker.attack_dataset(
        num_examples=100,
        log_to_csv="results/unconstrained_textfooler.csv",
        disable_stdout=True,
    )


if __name__ == "__main__":
    main()
