"""LLaMA-2 rewriting of adversarial candidates.

Original code from the project, with comments translated to English. Where
TextFooler swaps individual words for synonyms, this rewrites the whole
sentence and lets the classifier decide whether the rewrite still fools it.
"""

from transformers import pipeline

DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"


class AdversarialRefiner:
    """Refines an adversarial example with LLaMA-2 while keeping it adversarial.

    Any model that the `text-generation` pipeline accepts can be substituted;
    Llama-2-7b-chat is what the paper used. Loading in 8-bit keeps the model
    inside a single 24 GB card alongside the victim classifier.
    """

    def __init__(self, llama_model: str = DEFAULT_MODEL, device: int = 0, load_in_8bit: bool = True):
        self.pipe = pipeline(
            "text-generation",
            model=llama_model,
            device=device,
            model_kwargs={"load_in_8bit": load_in_8bit},
        )

    def refine_text(self, original: str, perturbed: str, target_label) -> str:
        prompt = (
            "Below is an original text and a perturbed text which is adversarial. "
            "We want to refine the perturbed text so it remains adversarial but stays coherent.\n\n"
            f"Original text: {original}\n\n"
            f"Perturbed text: {perturbed}\n\n"
            f"Target label: {target_label}\n\n"
            "Please refine the perturbed text. Only return the refined text."
        )
        result = self.pipe(prompt, max_new_tokens=128)
        return result[0]["generated_text"]
