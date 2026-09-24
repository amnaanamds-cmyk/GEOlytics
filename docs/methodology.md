# Experimental methodology

This document is the protocol. Follow it in order; several steps are only valid
if an earlier one was done first.

---

## 1. Corpus construction

Crawl 4–6 sites, 20–50 pages each. Use sites from different sectors so a
finding is not an artefact of one writing style.

* `robots.txt` is honoured, `Crawl-delay` respected, a floor delay applied on
  top, and the crawler identifies itself. State this in the report; it is an
  ethics requirement and an examiner may ask.
* Raw HTML is cached to `data/cache/`. **Do not clear it between experimental
  runs.** Re-crawling changes the corpus underneath your results and makes them
  unreproducible.
* Record the extractor that ran (`document.metadata["extractor"]`). Extractors
  disagree; a corpus built with two of them is not one corpus.

## 2. Judgment construction

```python
query_set, report = generate_query_set(documents, llm, QAGenerationConfig(...))
```

The generation segmentation (`DEFAULT_GENERATION_CHUNKER`) is **fixed and is
not one of the conditions under test**. If gold spans came from, say, the
semantic chunker's output, the semantic condition would be scored against
boundaries it chose for itself.

Report from `GenerationReport`:

| Quantity | Why it matters |
|---|---|
| `n_units`, `n_generated` | sample size |
| `n_units_too_short` | content excluded before generation. On a site of short sections this can be most of the page, leaving too few queries to test anything — lower `min_unit_tokens` rather than accepting it |
| `mean_lexical_overlap` | the bias toward BM25. Near 1.0 invalidates any claim that dense beats lexical |
| `n_dropped_overlap` | how much bias you removed |
| `n_failed_units` | LLM reliability |

### The template fallback is not the method

With no LLM configured, `build_query_set` falls back to
`heuristic_qagen.generate_heuristic_query_set`, which builds questions from
templates and corpus-salience terms. It exists so the pipeline runs from a
clean clone and in CI — not so the evaluation can skip the LLM.

Template questions reuse their source passage's vocabulary, which flatters BM25
and any hybrid retriever containing it. Queries are marked
`provenance="heuristic"`, the experiment output prints a warning, and
`QuerySet.filter_provenance` lets you separate them. Never report a result
produced from them.

### The step that defends everything else

1. Run all conditions, depth-pool the top-10 results (`build_pool`).
2. Sample ~100 queries. Label every pooled chunk by hand, **blind to the
   synthetic label**.
3. `judgment_agreement(synthetic, human, pool)` → Cohen's κ.
4. Re-run the headline comparison on the human-labelled subset alone.

If the two subsets rank the conditions the same way and κ is *moderate* or
better (≥ 0.41, Landis & Koch), the synthetic set is doing its job. This is the
single strongest answer to "but your ground truth is machine-generated".

κ rather than raw agreement: pools are dominated by irrelevant pairs, so two
labellings can agree 90% of the time while carrying no information.

## 3. Projection

```python
projected, stats = project_query_set(query_set, chunks, min_coverage=0.5)
```

Fix `min_coverage` **before** running anything, then re-run the headline
comparison at 0.3 and 0.7 as a sensitivity check. Report
`ProjectionStats.n_unmatched` per condition — a strategy that loses many
queries to projection failure is itself a finding.

## 4. Factors

| Factor | Levels | Status |
|---|---|---|
| Chunking | fixed, sentence, semantic, parent-child | **primary** |
| Retrieval | dense, BM25, hybrid (RRF), reranked | **primary** |
| Similarity metric | cosine, dot, Euclidean | **excluded — see below** |
| Embedding model | MiniLM, BGE, E5 | good replacement axis |
| Fusion weight α | 0.0 … 1.0 | good replacement axis (produces a curve) |

### Why the similarity metric is excluded

For L2-normalised vectors:

```
dot(a, b)   = cos(a, b)                    since ‖a‖ = ‖b‖ = 1
‖a − b‖²    = 2 − 2·dot(a, b)              expand the square
```

Euclidean distance is strictly decreasing in the dot product, so **ranking by
any of the three gives the same order**. Sentence-Transformers normalises by
default; Qdrant normalises on upsert for `Cosine` collections. Tabulating them
as three conditions produces three identical columns.

Run `geolytics check-metrics`, report τ = 1.0 and 100% identical top-k, and
present it as a short analytical result. The exception worth mentioning: models
trained for dot-product retrieval on *unnormalised* vectors (the
`multi-qa-*-dot-v1` family), where the vector norm encodes passage specificity.

## 5. Running the grid

```python
harness = ExperimentHarness(documents, query_set, embedder, top_k=10)
runs = harness.run_all(conditions)
comparisons = compare_runs(runs, metric="ndcg@10", baseline="fixed+dense")
```

Vary one factor at a time. Pre-register the primary comparison — the one the
conclusion rests on — before looking at any results.

Prefer a **baseline** to all-pairs: k−1 tests instead of k(k−1)/2 means a
gentler Holm correction and more power.

## 6. Statistics

Reported automatically by `compare_runs`:

* **Wilcoxon signed-rank**, not a paired t-test. Per-query retrieval metrics
  are bounded, discrete and pile up at 0 and 1; normality does not hold.
* **Percentile bootstrap 95% CI** on the mean paired difference, over 10,000
  resamples with a fixed seed (record it). The p-value says an effect exists;
  the interval says how big it plausibly is.
* **Cliff's δ** with Romano et al. (2006) thresholds.
* **Holm–Bonferroni** across the comparison family. Six uncorrected tests at
  α = 0.05 carry ~26% chance of a false positive. Correct within one family —
  one factor, one primary metric — not across unrelated metrics, which is
  over-conservative and hides real effects.

Paired tests run on the **intersection** of queries scored under every
condition (`common_query_ids`). Comparing means over different query subsets is
not a paired comparison.

**Always report `n chunks` and `mean tok`.** Chunk count and length are
confounders: many small chunks give more chances to hit and shorter passages to
match. `results_table` includes them by default.

## 7. GEO scoring

`geolytics calibrate <url>` runs steps 1–3; `geo.calibration.calibrate_weights`
is the library entry point.

1. Compute signals per page (`compute_signals`).
2. Run the simulated engine over the query set; collect
   `ImpressionMetrics.position_adjusted_share` per page, averaged over *every*
   query rather than only the ones where the page was cited.
3. `fit_weights(reports, observed)` — non-negative least squares.
4. Report R², `n_observations`, and `signal_correlations`.

Calibration refuses to fit — and says why — when there are no more pages than
signals (the system is underdetermined) or when fewer than two pages were cited
(the regression target has no variance). Both cases return uniform weights
flagged `fitted=False`, which the scorer and the API then surface as a caveat
rather than a number.

Non-negative because a negative coefficient would claim "adding citations makes
you less visible", which this data cannot support and which cannot be explained
to a site owner.

**Never present a score from unfitted weights as a finding.** The default
weights are uniform and flagged; the API returns an explicit caveat string.

## 8. External validation

The step that makes the project genuinely defensible.

1. Take ~50 queries from the evaluation set.
2. Run each manually on Perplexity and Google AI Overviews. Record whether the
   audited site appears, and where. Record the date — these systems change.
3. Correlate observed appearance against the GEO score / predicted visibility
   (Spearman ρ, since appearance is ordinal).
4. Report the correlation with its CI and n.

Phrase the conclusion exactly like this:

> This work does not claim causal attribution of any generative engine's
> internal ranking. It reports that the simulated visibility score correlates
> with observed appearance in commercial generative engines at ρ = 0.5x
> (n = 50, 95% CI […]), measured on <date>.

## Threats to validity — write these up, do not hide them

| Threat | Mitigation in the system |
|---|---|
| Synthetic queries are lexically biased toward their source | `lexical_overlap` measured per query, threshold drop, paraphrase instruction, human-κ validation |
| Chunk-level judgments are not portable across chunkers | span anchoring + projection |
| Chunk size confounds retrieval quality | `IndexStats` reported in every table |
| A local 7–8B model is a noisy judge | validate against a human-labelled sample and report agreement; never present judge scores as ground truth |
| Multiple comparisons inflate false positives | Holm–Bonferroni per family |
| Fitted weights are associations, not causes | stated at every point the weights appear; the causal version is an intervention study |
| ANN recall loss confounds the index | experiments use exact search; measure the Qdrant gap rather than assuming it away |
| Regex sentence segmentation | swap in spaCy/PySBD **before** the experiments, never between them |
| URL aliases duplicating content | the crawler fingerprints extracted text and drops repeats; `skipped_duplicate` is reported |
| Template-generated queries leaking into results | marked `provenance="heuristic"`; filter with `QuerySet.filter_provenance` |
| Generative engines change over time | record the date of every external observation |

## Suggested milestones

| Weeks | Deliverable |
|---|---|
| 1–2 | Crawler, extraction, caching |
| 3–4 | Chunking, embedding, Qdrant indexing |
| 5–6 | Retrieval variants, synthetic QA generation |
| 7–8 | Evaluation harness, metrics, paired statistics |
| 9–10 | GEO signals, simulated engine, weight fitting |
| 11 | Human labelling (~100 queries), κ, external validation |
| 12–13 | Dashboard |
| 14 | Report |

Week 11 is the one students skip and the one that carries the viva. Do not cut
it — cut dashboard polish instead.

## References to verify before citing

- Aggarwal et al., *GEO: Generative Engine Optimization*, KDD 2024 — verify the
  exact impression-metric definitions and reported lift figures.
- Cormack et al., *Reciprocal Rank Fusion*, SIGIR 2009.
- Landis & Koch (1977) — κ interpretation benchmarks.
- Romano et al. (2006) — Cliff's δ thresholds.
- Holm (1979) — the step-down correction.
