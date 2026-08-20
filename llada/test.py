#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""GSM8K prompt construction and scoring utilities (shared by the LLaDA harness).

Only prompt building and the scoring helpers are kept here. Evaluation/generation
for LLaDA is a diffusion sampling loop and lives in ``train.py`` (it cannot use
the autoregressive ``model.generate()`` path the original GSM8K ``test.py`` used).
"""

import re


ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


def build_prompt(tokenizer, question: str) -> str:
    """Render one GSM8K question in the model's own chat format.

    LLaDA-8B-Instruct is instruction-tuned inside a Llama-3-style chat template
    (``<|start_header_id|>user<|end_header_id|>`` ... ``<|eot_id|>`` ...
    ``<|start_header_id|>assistant<|end_header_id|>``), and LLaDA's reference
    sampler applies it before every generation. Feeding the bare question instead
    runs the model outside the format it was instruction-tuned in, and published
    LLaDA numbers are not comparable to anything measured that way.

    The template is read off the tokenizer rather than hardcoded here, so whatever
    the model repo ships is what gets used.

    Both the SFT targets (``train.SupervisedDataset``) and the benchmark
    (``train.diffusion_evaluate``) build their prompts through this function. They
    have to agree exactly: a mismatch scores every adapter outside the format it
    was trained in, which looks like a bad adapter rather than a bad harness.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": f"{question}{QUESTION_PROMPT}"}],
        add_generation_prompt=True,
        tokenize=False,
    )


def template_add_special_tokens(tokenizer) -> bool:
    """Should ``tokenizer(build_prompt(...))`` be allowed to add special tokens?

    ``apply_chat_template(tokenize=False)`` renders ``bos_token`` into the string
    it returns. A tokenizer that *also* prepends BOS when called would then emit
    it twice -- a prefix the model has never seen at training time. Detect that by
    rendering a throwaway prompt and asking whether the BOS is already there,
    rather than assuming either behaviour of LLaDA's custom tokenizer.

    Returned as a flag rather than wrapped up in a tokenize helper because
    ``train.preprocess`` tokenizes prompt+target concatenated as well as the
    prompt alone, and must pass the same flag to both or the prompt-length label
    mask goes out of alignment.
    """
    bos = getattr(tokenizer, "bos_token", None)
    if not bos:
        return True
    return not build_prompt(tokenizer, "probe").startswith(bos)


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError:
            pred_answer = float('inf')
    return pred_answer


def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)
