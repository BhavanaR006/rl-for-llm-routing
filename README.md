# Cost-Aware LLM Routing with Online Reinforcement Learning

For each query, decide which of five open language models should answer it, so that cheap models are used when they are good enough.
This repository compares a router that learns online (a contextual bandit) against HybridLLM, FrugalGPT, a RouterBench-style predictive router and GraphRouter, on two datasets.

**Short on time?** Open the latest deck: [`03_rl_routing/slides/02_rl_vs_routerbench_graphrouter.pptx`](03_rl_routing/slides/02_rl_vs_routerbench_graphrouter.pptx) (RL vs RouterBench-style and GraphRouter routers). The earlier deck is [`01_rl_vs_hybridllm_frugalgpt.pptx`](03_rl_routing/slides/01_rl_vs_hybridllm_frugalgpt.pptx). Notebooks are saved with their outputs, so every result can be read without rerunning anything.

## How the project developed

The work was done in four phases, and the folders and notebook numbers follow the same order.

1. **`01_frugalgpt/`: understand FrugalGPT.** I started with the FrugalGPT paper (Chen, Zaharia, Zou, TMLR 2024) and reproduced its cascade idea on the HEADLINES dataset: query the cheapest model first and let a learned scorer decide whether to escalate to a stronger one.
2. **`02_hybridllm/`: implement HybridLLM.** Next I implemented HybridLLM (Ding et al., ICLR 2024), which trains a router offline to choose between two models. I first reproduced the original two-model setup (FLAN-T5 and Llama-2-7b on MixInstruct), then extended it to pairs drawn from the five models used in this project and estimated its cost.
3. **`03_rl_routing/` (notebooks 01 to 06): build an RL router and compare it.** I built a router that works with five models and learns online (a contextual bandit), and compared it with HybridLLM and FrugalGPT on two datasets. The later notebooks add multiple seeds and seven exploration rules, following feedback that a single run is not reliable.
4. **`03_rl_routing/` (notebook 07): compare with RouterBench-style and GraphRouter routers.** Following the RouterBench paper (Hu et al., 2024) and a search for recent accepted routing papers, I added two more routers on the same data and cost, and scored them with RouterBench's Zero-router and AIQ ideas.

## Setup

- **Models (cost = parameters / 7B):** flan-t5-small (80M, 0.011), flan-t5-base (250M, 0.036), flan-t5-large (780M, 0.111), flan-t5-xl (3B, 0.429), Llama-2-7b-chat (7B, 1.0). "Cost" in every table is the average cost per test query of the model the router picked.
- **Datasets:** MMLU (accuracy; 1,531 test questions) and MixInstruct (pass rate = BARTScore >= -4.55; 1,500 test queries).
- **Score used for every method ("utility"):** (+1 if right, -1 if wrong) - 0.5 x cost. Higher is better.
- **Methods:**
  - *Single model:* always use one model.
  - *RL / bandit router (mine):* reads the question, picks one model, and learns online from the reward (+1 / -1 minus lambda x cost). It only sees the reward of the model it picked, so the rules differ in how they explore (see below).
  - *HybridLLM* (Ding et al., ICLR 2024): a router trained offline to choose between two models; three pairs (small / large / xl + Llama-2).
  - *FrugalGPT* (Chen, Zaharia, Zou, TMLR 2024): a cascade, cheapest model first, with a learned scorer deciding whether to escalate. Run on MMLU only.
  - *Predictive router (RouterBench-style):* an MLP predicts the probability that each model answers correctly, and the router picks the model with the best (2P - 1) - lambda x cost. Same recipe as the RouterBench predictive routers, and as the quality head of R2-Router (ICML 2026) without its output-length axis, so it is not called R2-Router here.
  - *GraphRouter* (Feng, Shen, You, ICLR 2025): a graph of task, query and LLM nodes with a 2-layer graph network that predicts the best model per query. This is my re-implementation on the same features (see Limitations).
  - *Zero router and AIQ* (RouterBench): the Zero router mixes single models along their cost-quality line; a router is useful only if it beats it. AIQ is the area under a router's cost-quality curve.
- **Protocol:** lambda and bandit parameters are chosen on a separate validation split; results are reported on the test set.
- **Why not a 2026 method:** R2-Router (ICML 2026) needs answers collected under several output-length budgets, and RouterArena (ICLR 2026) and LLMRouterBench (Findings of ACL 2026) are benchmarks. None can run on tables that hold one answer per model per question, so an accepted 2025 paper (GraphRouter) was used.

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
| Best routed RL policy (frozen bandit chosen on validation, 20 seeds) | LinUCB (alpha 0.5): 46.1% at cost 0.234, utility -0.195 | LinTS (v 0.5): 86.7% at cost 0.592, utility +0.437 |
| Gain over best single model | +0.020 | +0.118 |

- Routing helps most when the models differ clearly (MixInstruct). On MMLU the cheap models are already close to the best.
- One seed can mislead: fine-tuned epsilon-greedy on MixInstruct (lambda 0.5) ended with cost 0.59 to 0.90 across three seeds.
- The exploration rule matters little: the seven rules differ by at most 0.012 (MMLU) and 0.029 (MixInstruct) in utility. Boltzmann matched or beat epsilon-greedy on all 6 fine-tuned seed pairs (strictly on 5), which is only indicative with 3 seeds per dataset.
- Compute to learn, per training question: RL about 1.8B (MMLU) / 4.4B (MixInstruct) parameter-calls, HybridLLM 7.1 to 10.0B, FrugalGPT cascade 11.1B (MMLU only).

### RL vs RouterBench-style and GraphRouter routers (notebook 07, 5 seeds)

| Method | MMLU cost / accuracy / utility | MixInstruct cost / pass rate / utility |
|---|---|---|
| Best single model | 0.111 / 42.0% / -0.216 | 1.000 / 90.9% / +0.319 |
| Predictive router (RouterBench-style) | 0.207 / 46.1% / -0.182 +- 0.011 | 0.502 / 86.6% / +0.481 +- 0.008 |
| GraphRouter (ICLR 2025, re-implemented) | 0.085 / 35.2% / -0.339 +- 0.025 | 0.288 / 73.3% / +0.323 +- 0.014 |
| RL, best fine-tuned row (3 seeds) | 0.226 / 46.3% / -0.187 | 0.643 / 89.2% / +0.464 |
| RL, frozen bandit (20 seeds) | 0.234 / 46.1% / -0.195 | 0.592 / 86.7% / +0.437 |

- The predictive router ties RL on MMLU and leads it on MixInstruct by 0.017 to 0.044 utility, at lower cost. RL is not the best on quality here.
- RL's advantage is the cost of learning: the predictive router and GraphRouter need every model's answer for every training question (about 11.1B parameter-calls each), while RL needs 1.8B to 4.4B and feedback only on the answer it served.
- Against the Zero router at the same cost, the predictive router is ahead by about 1.7 points on MMLU and 14.8 points on MixInstruct. AIQ on MixInstruct: 0.818 (predictive) vs 0.720 (Zero router).
- GraphRouter falls below the best single model on MMLU and only matches it on MixInstruct. With right-or-wrong scores its best-model labels are identical for every lambda below 2, so the result is not evidence about GraphRouter in general.

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
    slides/01_rl_vs_hybridllm_frugalgpt.pptx        presentation 1
    slides/02_rl_vs_routerbench_graphrouter.pptx    presentation 2 (latest)
    slides/archive/                          earlier decks, kept for history
```

## Reading order (`03_rl_routing/notebooks/`)

1. `01_rl_router_mixinstruct.ipynb`: first RL router on MixInstruct with live generation and BARTScore.
2. `02_rl_vs_hybridllm_mixinstruct.ipynb`: RL vs HybridLLM (3 pairs) on MixInstruct, matched-cost comparison with bootstrap intervals.
3. `03_rl_vs_hybridllm_mmlu.ipynb` and `03_rl_vs_hybridllm_mmlu_lambda_sweep.ipynb`: the same comparison on MMLU, with a sweep of the cost weight lambda.
4. `04_rl_vs_frugalgpt_mmlu.ipynb`: RL vs the FrugalGPT cascade on MMLU.
5. `05_multiseed_bandits_mmlu.ipynb` and `05_multiseed_bandits_mixinstruct.ipynb`: the same notebook run once per dataset (`DATASET` in the config cell); multi-seed bandits, Stage A (frozen features, 20 seeds) and Stage B (fine-tuned, 3 seeds).
6. `06_cross_dataset_comparison.ipynb`: reads the two multi-seed result files and compares MMLU with MixInstruct.
7. `07_rl_vs_graphrouter_mmlu.ipynb` and `07_rl_vs_graphrouter_mixinstruct.ipynb`: the same notebook run once per dataset; trains the predictive router and GraphRouter on the same tables and features and compares them with RL, HybridLLM and FrugalGPT, with Zero-router and AIQ scoring.

## Running the code

The notebooks were run on a shared GPU server (Jupyter, Python 3.10, one NVIDIA L40S). Intermediate results are cached under `~/hybridllm_project/` (`ckpt_shared/` and `outputs/`), and later notebooks read the caches written by earlier ones (notebooks 02 and 03 before 05, and 05 before 06 and 07). To run elsewhere, edit `WORK_DIR` in each notebook's setup cell. `hybridllm_original_pair_kaggle.ipynb` was run on Kaggle.


Saved result files are in 03_rl_routing/results/

```bash
pip install -r requirements.txt
jupyter lab
```

Models and datasets are downloaded from Hugging Face (`google/flan-t5-*`, `NousResearch/Llama-2-7b-chat-hf`, `microsoft/deberta-v3-small`, `facebook/bart-large-cnn`, `cais/mmlu`, `llm-blender/mix-instruct`). No model weights are stored here, and the Llama-2 license applies to its use.

## Limitations

- HybridLLM and the FrugalGPT cascade are single runs, so they have no error bars; FrugalGPT was not run on MixInstruct.
- Utility uses a fixed cost weight of 0.5; other weights were not tested.
- The RL reward is read from a table of precomputed answers, so the "online" feedback is simulated; real use would need a signal such as a thumbs up or down.
- GraphRouter is a simplified re-implementation: learned task and LLM vectors instead of text embeddings, PCA features, batch 256 and 6,000 steps. Its labels degenerate with binary correctness, as noted above.
- The predictive router and GraphRouter are offline and use every model's answer for every training question, which RL never sees, so their quality is not a like-for-like comparison with RL.
- Intervals over seeds reflect learning randomness only, not sampling of the test questions. GPU runs were not forced to be deterministic.

## References

- Ding et al., *Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*, ICLR 2024 (arXiv:2404.14618).
- Chen, Zaharia, Zou, *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, TMLR 2024 (arXiv:2305.05176).
- Hu et al., *RouterBench: A Benchmark for Multi-LLM Routing System*, 2024 (arXiv:2403.12031).
- Feng, Shen, You, *GraphRouter: A Graph-based Router for LLM Selections*, ICLR 2025 (arXiv:2410.03834).
- Li et al., *A Contextual-Bandit Approach to Personalized News Article Recommendation* (LinUCB), WWW 2010.
- Agrawal and Goyal, *Thompson Sampling for Contextual Bandits with Linear Payoffs*, ICML 2013.
- Hendrycks et al., *Measuring Massive Multitask Language Understanding* (MMLU), ICLR 2021.
- Jiang et al., *LLM-Blender* (MixInstruct), ACL 2023.
