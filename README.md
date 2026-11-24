# Learning from Failed Hypotheses in HypoSpace

EEC 4300 final project. A language model repeatedly proposes explanations that fit incomplete observations. We study whether an inference-time decoder can use the reason a candidate failed to improve the next proposal. The decoder learns partial graph exclusions from failed candidates and rules out repeated hypotheses. We evaluate it on the 3D voxel and causal perturbation tasks from HypoSpace. [HypoSpace paper](https://proceedings.mlr.press/v306/chen26bz.html)

## Method

Each hypothesis has a canonical encoding. For voxels, it is the sequence of occupied column heights in row-major order. For causal graphs, it is the bit vector of all directed edges between distinct labeled nodes. A grammar describes the encodings the model may generate. At each query, the model samples from this grammar and a deterministic validator inspects its answer.

| Feedback | What the next grammar excludes |
| --- | --- |
| Any already tried encoding | Exactly that encoding, including any invalid encoding seen so far. |
| Directed cycle | Every graph containing all edges of that cycle. |
| Unobserved effect | Every graph containing the witnessed path to the unobserved node. |
| Missing effect | Every graph that keeps all edges from the reachable set to its complement absent. |

The proposed decoder uses a complete exclusion for each valid answer and a partial exclusion for each failed graph with an identified cause. A separate control excludes complete answers only. The compiler tracks which forbidden partial assignments are still consistent with the current output prefix, drops contradicted assignments and blocks a branch as soon as it satisfies a complete forbidden assignment. Equivalent states share grammar rules. The language model picks the remaining digits. Its weights are frozen.

### Why the exclusions are sound

For an intervention at node s, let S be s together with all observed descendants. An edge from S to its complement would expose an unobserved effect in any compatible graph, so the decoder can fix those edge bits to zero before sampling. The remaining constraints are checked on complete proposals.

A cycle and a path to an unobserved effect remain present in any graph containing the witnessed edges. For a missing effect, let R be s together with every node reachable in the proposed graph. A required node lies outside R. Every directed path from s to that node crosses from R to its complement at least once. If all such edges are absent, the required node remains unreachable. The missing-effect certificate records the zeros for those cut edges that the static grammar still permits. Therefore every learned partial exclusion rejects only invalid graphs within the static grammar. Blocking previously emitted valid encodings prevents duplicates while preserving every unseen valid hypothesis.

The model receives the observations, the encoding definition, recent valid outputs and the current grammar. The reference count is computed independently and used only for case selection and scoring.

### Relation to prior work

- HypoSpace evaluates valid, diverse explanations and studies complexity-stratified prompts. Our first experiment compares fixed and adaptive height-prefix prompting with that approach. The main experiment moves the constraints into the decoder and tests whether a failed proposal can rule out a family of graphs. [Chen et al., ICML 2026](https://proceedings.mlr.press/v306/chen26bz.html)
- Input-dependent grammars constrain what a language model can emit. Stochastic beam search can sample distinct sequences from a factorized model. Our grammar changes after each verifier response, so the comparisons concern search coverage rather than preservation of the original model distribution. [Geng et al., EMNLP 2023](https://aclanthology.org/2023.emnlp-main.674/) [Kool et al., ICML 2019](https://proceedings.mlr.press/v97/kool19a.html)
- The use of partial exclusions is inspired by conflict learning in SAT. The project contribution is the reachability certificates and their use in repeated LLM sampling for these tasks. Semantic constraints during decoding have also been studied in systems such as ChopChop. [Marques-Silva and Sakallah, GRASP](https://www.eecs.umich.edu/courses/eecs578/f15/papers/mar99.pdf) [ChopChop](https://doi.org/10.1145/3776708)

## Evaluation

The main experiment pairs all methods on the same 24 observations and uses 30 proposals per method and observation. It repeats the design with Qwen2.5-1.5B-Instruct and SmolLM2-1.7B-Instruct, both in Q4_K_M. Within each model, methods share decoding settings, initial seeds, answer encodings, and the same policy for retaining up to 12 recent valid answers in the prompt. One attempted proposal consumes one call.

| Domain | Instances | Structure space |
| --- | ---: | --- |
| 3 × 3 voxel top view, three occupied cells | 6 | 125 structures per observation |
| 3 × 3 voxel top view, four occupied cells | 6 | 625 structures per observation |
| Four labeled causal nodes, one intervention | 6 | 40–79 matching DAGs per observation |
| Five labeled causal nodes, two interventions | 6 | 277–576 matching DAGs per observation |

The voxel dimensions use the benchmark's gravity and top-view definitions. The causal task matches exact descendant vectors on synthetic graphs, with independently sampled observations and no extra edge-count cap. It uses the same task and metrics as HypoSpace with a compact encoding, a fixed call budget and selected sizes. Published scores use different observation sets, prompts and budgets. [Official generator and runner](https://github.com/CTT-Pavilion/_HypoSpace)

| Method | What is constrained |
| --- | --- |
| Memory | Canonical output syntax and recent valid answers in the prompt. |
| Fixed prefix | In the voxel task, cycles through seeded pairs of first and second heights, enforced by the grammar. |
| Static | In the causal task, adds zero edges implied by observed unaffected nodes. |
| Exact exclusion | With the static grammar where applicable, blocks each full encoding after its first occurrence. |
| Learned exclusion | In the causal task, generalizes each failure to an edge pattern and blocks each valid encoding after its first occurrence. |

Validity is the fraction of calls with a compatible hypothesis. Uniqueness is the fraction producing a distinct parseable hypothesis, including invalid ones. Recovery is the fraction of all compatible hypotheses found. We also report distinct valid hypotheses after 5, 10, 20 and 30 calls, token costs, counts of failed proposals by cause, and paired 95% bootstrap intervals stratified by setting. These intervals reflect variation across the sampled observations conditional on the models and decoding seeds.

The no-model control samples uniformly from the same current grammar with exact exclusions or learned exclusions. Each eligible next digit is weighted by the number of completions below it. We average five seeded runs on each causal observation. This tests whether the improvement can arise from the verifier and constraints alone.

## Results

### Causal graph discovery

On both models, learned exclusions increased the number of distinct valid graphs from about six or seven to more than 23 per observation in 30 calls. Exact exclusion and learned exclusion both achieved 100% Uniqueness. The improvement came from fewer invalid proposals.

| Model | Method | Validity | Uniqueness | Recovery | Distinct valid graphs |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen | Memory | 10.00% | 81.94% | 0.75% | 0.42 |
| Qwen | Static | 30.83% | 51.94% | 2.62% | 1.42 |
| Qwen | Exact exclusion | 21.11% | 100.00% | 10.18% | 6.33 |
| Qwen | Learned exclusion | **78.06%** | **100.00%** | **27.98%** | **23.42** |
| SmolLM2 | Memory | 9.17% | 85.83% | 0.92% | 0.50 |
| SmolLM2 | Static | 29.72% | 55.56% | 3.18% | 1.75 |
| SmolLM2 | Exact exclusion | 23.06% | 100.00% | 9.51% | 6.92 |
| SmolLM2 | Learned exclusion | **77.50%** | **100.00%** | **27.46%** | **23.25** |

Compared with exact exclusion, the paired mean Recovery gain was **17.80 percentage points** for Qwen (95% bootstrap interval 11.49 to 24.54) and **17.95 points** for SmolLM2 (12.73 to 22.74). Recovery averages the per-observation fractions. A given number of valid discoveries contributes less Recovery on a larger reference space.

| Model and method | At 5 calls | At 10 calls | At 20 calls | At 30 calls |
| --- | ---: | ---: | ---: | ---: |
| Qwen, exact | 0.50 | 1.58 | 4.08 | 6.33 |
| Qwen, learned | 1.92 | 5.42 | 13.83 | 23.42 |
| SmolLM2, exact | 0.50 | 1.17 | 3.92 | 6.92 |
| SmolLM2, learned | 1.17 | 5.00 | 13.83 | 23.25 |

Both models returned an average of 13.83 distinct valid graphs after 20 calls with learned exclusion. The longer memory of valid results increased average input tokens per query from 182 to 273 for Qwen and from 189 to 276 for SmolLM2. Output token counts were equal within each model. The learned grammar used fewer rules on average than exact exclusion, 57 versus 70 for Qwen and 60 versus 72 for SmolLM2. These are measurements of query and rule size on the tested graphs.

On the five-node instances alone, learned exclusion found 23.50 valid graphs on average for each model. Exact exclusion found 3.33 for Qwen and 5.50 for SmolLM2. The larger spaces contain 277 to 576 valid graphs per observation, so the same 30-call budget still covers only a small share of the admissible space.

With uniform proposals and no model, learned exclusion increased the mean discoveries from **3.50** to **22.20** and mean Recovery from **4.79%** to **27.07%**. The symbolic feedback accounts for much of the observed improvement. Under learned exclusion, the uniform control and the two language models achieved similar mean Recovery.

### Voxel reconstruction

| Model | Memory | Fixed prefix | Exact exclusion |
| --- | ---: | ---: | ---: |
| Qwen | 15.83 | 28.75 | **30.00** |
| SmolLM2 | 11.17 | 28.83 | **30.00** |

The table reports mean distinct valid structures after 30 calls. Both models reached 30 on every footprint with exact exclusion, the maximum at one proposal per call. Validity was 100% for all methods because the shared height grammar enforces the task geometry. Exact exclusion also reached 100% Uniqueness, 24.00% Recovery in the 125-solution setting, and 4.80% Recovery in the 625-solution setting. A direct sampler without replacement reaches the same ceiling. This task verifies the duplicate-blocking mechanism in a known finite space.

### Scope

The main experiment consists of 5,040 model calls, 2,520 per model. Results apply to two small quantized models and the four synthetic settings evaluated. Exact solvers remain available on these small instances. The online grammar is efficient on the observed cases, while the number of grammar states can grow with the variety and overlap of learned exclusions. Applying the method to a new domain requires a validator that can identify a sound partial cause of failure.

## Reproduce

Use Python 3.10 or later for the scripts. There are no third-party Python dependencies. Run [llama.cpp b12674](https://github.com/ggml-org/llama.cpp/releases/tag/b12674) with the [Qwen model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF) or the [SmolLM model](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF). Launch each model in its own PowerShell window. Run the following from the executable and model directory, with GPU flags suited to the build and available hardware.

```powershell
.\llama-server.exe --model .\qwen2.5-1.5b-instruct-q4_k_m.gguf --alias qwen2.5-1.5b-instruct-q4_k_m --ctx-size 8192 --parallel 8 --host 127.0.0.1 --port 18385
.\llama-server.exe --model .\smollm2-1.7b-instruct-q4_k_m.gguf --alias smollm2-1.7b-instruct-q4_k_m --ctx-size 8192 --parallel 8 --host 127.0.0.1 --port 18386
```

From the project directory, use fresh transcripts for independent runs or a transcript with the same configuration to resume. The reported runs offloaded both models to an RTX 4060 Laptop GPU using Vulkan.

```powershell
python -B benchmark.py run --model qwen2.5-1.5b-instruct-q4_k_m --output results/reproduction-qwen.jsonl
python -B benchmark.py run --endpoint http://127.0.0.1:18386/v1 --model smollm2-1.7b-instruct-q4_k_m --output results/reproduction-smollm.jsonl
```

These commands reconstruct the cases, replay the validator and grammar decisions, and recompute metrics without a model server.

```powershell
python -B benchmark.py score results/constraint-qwen.jsonl --uniform-repeats 5
python -B benchmark.py score results/constraint-smollm.jsonl
python -B -m unittest discover -v
```

The first experiment on height-prefix prompting is retained in results/qwen25-1.5b.jsonl and results/qwen25-1.5b-format.jsonl. Its runner is experiment.py and its scorer is summarize.py. Both ran 24 voxel observations using Qwen under free-format or common integer-array output constraints. The two prefix schedules had similar Recovery. Under the shared integer-array constraint, they achieved 32.65% for fixed prefix and 32.62% for adaptive prefix, compared with 14.12% for memory and 19.10% for complexity stratification.

