# Routing queries across five LLMs: RL / bandit routers vs HybridLLM vs FrugalGPT

For each query, decide which of five open language models should answer it, so that cheap models are used when they are good enough.
This repository compares a router that learns online (a contextual bandit) against two published methods, on two datasets.

**Short on time?** Open the final deck: [`03_rl_routing/slides/routing_comparison_final.pptx`](03_rl_routing/slides/routing_comparison_final.pptx). Notebooks are saved with their outputs, so every result can be read without rerunning anything.

## How the project developed

The work was done in three phases, and the folders follow the same order.

1. **`01_frugalgpt/`: understand FrugalGPT.** I started with the FrugalGPT paper (Chen, Zaharia, Zou, TMLR 2024) and reproduced its cascade idea on the HEADLINES dataset: query the cheapest model first and let a learned scorer decide whether to escalate to a stronger one.
2. **`02_hybridllm/`: implement HybridLLM.** Next I implemented HybridLLM (Ding et al., ICLR 2024), which trains a router offline to choose between two models. I first reproduced the original two-model setup (FLAN-T5 and Llama-2-7b on MixInstruct), then extended it to pairs drawn from the five models used in this project and estimated its cost.
3. **`03_rl_routing/`: build an RL router and compare everything.** Finally I built a router that works with five models and learns online (a contextual bandit), and compared it with HybridLLM and FrugalGPT on two datasets. The later notebooks add multiple seeds and seven exploration rules, following feedback that a single run is not reliable.

## Setup

- **Models (cost = parameters / 7B):** flan-t5-small (80M, 0.011), flan-t5-base (250M, 0.036), flan-t5-large (780M, 0.111), flan-t5-xl (3B, 0.429), Llama-2-7b-chat (7B, 1.0).
- **Datasets:** MMLU (accuracy; 1,531 test questions) and MixInstruct (pass rate = BARTScore >= -4.55; 1,500 test queries).
- **Score used for every method ("utility"):** (+1 if right, -1 if wrong) - 0.5 x cost. Higher is better.
- **Methods:**
  - *Single model:* always use one model.
  - *RL / bandit router:* reads the question, picks one model, and learns online from the reward (+1 / -1 minus lambda x cost). It only sees the reward of the model it picked, so the rules differ in how they explore (see below).
  - *HybridLLM* (Ding et al., ICLR 2024): a router trained offline to choose between two models; three pairs (small / large / xl + Llama-2).
  - *FrugalGPT* (Chen, Zaharia, Zou, TMLR 2024): a cascade, cheapest model first, with a learned scorer deciding whether to escalate. Run on MMLU only in the comparison.
- **Protocol:** lambda and bandit parameters are chosen on a separate validation split; results are reported on the test set.

### The two RL experiments (Stage A and Stage B)

| | Stage A | Stage B |
|---|---|---|
| Question features | DeBERTa-v3-small, frozen (PCA to 64 numbers + bias) | DeBERTa-v3-small, fine-tuned while learning |
| Exploration rules | 7: epsilon-greedy and Boltzmann (small MLP), epsilon-greedy (linear), LinUCB x2, linear Thompson sampling x2 | 2: epsilon-greedy and Boltzmann |
| Runs per dataset | 7 rules x 5 lambdas x 20 seeds = 700 | 2 rules x 2 lambdas x 3 seeds = 12 |
| Cost of one run | cheap (CPU) | 6 to 12 minutes (GPU) |

Stage A is cheap, so many rules and seeds fit. Stage B is the real system, but slow, so only the two rules that train the same neural-network head were kept, with 3 seeds.

A **seed** fixes the order of the training questions, the starting weights of the network and the random exploration choices. It does not change the train / validation / test split, the features or the table of each model's right and wrong answers. The same seed gives every rule the same question order, so comparisons are paired.

## Headline results (test set)

| | MMLU | MixInstruct |
|---|---|---|
| Best single model (chosen on validation) | flan-t5-large, utility -0.216 | Llama-2-7b-chat, utility +0.319 |
| Best routed policy (frozen bandit chosen on validation, 20 seeds) | LinUCB (alpha 0.5): 46.1% at cost 0.234, utility -0.195 | LinTS (v 0.5): 86.7% at cost 0.592, utility +0.437 |
| Gain over best single model | +0.020 | +0.118 |

- Routing helps most when the models differ clearly (MixInstruct). On MMLU the cheap models are already close to the best.
- One seed can mislead: fine-tuned epsilon-greedy on MixInstruct (lambda 0.5) ended with cost 0.59 to 0.90 across three seeds.
- The exploration rule matters little: the seven rules differ by at most 0.012 (MMLU) and 0.029 (MixInstruct) in utility. Boltzmann matched or beat epsilon-greedy on all 6 fine-tuned seed pairs (strictly on 5), which is only indicative with 3 seeds per dataset.
- Compute to learn, per training question: RL about 1.8B (MMLU) / 4.4B (MixInstruct) parameter-calls, HybridLLM 7.1 to 10.0B, FrugalGPT cascade 11.1B (MMLU only).

## Repository layout

```
01_frugalgpt/
    frugalgpt_headlines_3providers_2000.ipynb
    frugalgpt_headlines_5services_2000.ipynb
    frugalgpt_headlines_full10000.ipynb
    slides/frugalgpt_explainer.pptx
02_hybridllm/
    hybridllm_original_pair_kaggle.ipynb     original two-model reproduction (run on Kaggle)
    hybridllm_5models.ipynb                  HybridLLM on pairs of the five models
    hybridllm_cost_estimate_pairs.ipynb      analytical cost of three model pairs
    results/                                 hybridllm_results.json, hybridllm_implementation.png
    slides/hybridllm_explainer.pptx
03_rl_routing/
    notebooks/                               numbered in reading order (below)
    slides/routing_comparison_final.pptx     final deck
    slides/archive/                          earlier decks, kept for history
```

## Reading order (`03_rl_routing/notebooks/`)

1. `01_rl_router_mixinstruct.ipynb`: first RL router on MixInstruct with live generation and BARTScore.
2. `02_rl_vs_hybridllm_mixinstruct.ipynb`: RL vs HybridLLM (3 pairs) on MixInstruct, matched-cost comparison with bootstrap intervals.
3. `03_rl_vs_hybridllm_mmlu.ipynb` and `03_rl_vs_hybridllm_mmlu_lambda_sweep.ipynb`: the same comparison on MMLU, with a sweep of the cost weight lambda.
4. `04_rl_vs_frugalgpt_mmlu.ipynb`: RL vs the FrugalGPT cascade on MMLU.
5. `05_multiseed_bandits_mmlu.ipynb` and `05_multiseed_bandits_mixinstruct.ipynb`: the same notebook run once per dataset (`DATASET` in the config cell); multi-seed bandits, Stage A (frozen features, 20 seeds) and Stage B (fine-tuned, 3 seeds).
6. `06_final_comparison.ipynb`: reads the two multi-seed result files and compares MMLU with MixInstruct.

## Running the code

The notebooks were run on a shared GPU server (Jupyter, Python 3.10, one NVIDIA L40S). Intermediate results are cached under `~/hybridllm_project/` (`ckpt_shared/` and `outputs/`), and later notebooks read the caches written by earlier ones (steps 3 and 2 before step 5). To run elsewhere, edit `WORK_DIR` in each notebook's setup cell. `hybridllm_original_pair_kaggle.ipynb` was run on Kaggle.

```bash
pip install -r requirements.txt
jupyter lab
```

Models and datasets are downloaded from Hugging Face (`google/flan-t5-*`, `NousResearch/Llama-2-7b-chat-hf`, `microsoft/deberta-v3-small`, `facebook/bart-large-cnn`, `cais/mmlu`, `llm-blender/mix-instruct`). No model weights are stored here, and the Llama-2 license applies to its use.


## References

- Ding et al., *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*, ICLR 2024 (arXiv:2404.14618).
- Chen, Zaharia, Zou, *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, TMLR 2024 (arXiv:2305.05176).
- Li et al., *A Contextual-Bandit Approach to Personalized News Article Recommendation* (LinUCB), WWW 2010.
- Agrawal and Goyal, *Thompson Sampling for Contextual Bandits with Linear Payoffs*, ICML 2013.
- Hendrycks et al., *Measuring Massive Multitask Language Understanding* (MMLU), ICLR 2021.
- Jiang et al., *LLM-Blender* (MixInstruct), ACL 2023.
