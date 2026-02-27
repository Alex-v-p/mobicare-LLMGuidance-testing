# 005 Graph Augmentation

This experiment evaluates *graph-augmented retrieval* on the guideline document.

## What is "graph augmentation" here?
We first do normal vector retrieval (top-k by cosine similarity).  
Then we expand the candidate set by walking a pre-built **chunk–chunk similarity graph** (k-NN graph over chunk embeddings).
Finally, we **re-rank** the expanded candidates with a mixture of:
- query-to-chunk similarity
- graph proximity to the initially retrieved chunks

This simulates GraphRAG-style “neighbor expansion” without introducing an LLM and keeps the evaluation comparable to the earlier retrieval tests.

## Outputs
- `leaderboard_graph_aug.csv`: one row per run (baseline + augmented metrics + deltas)
- `results/runs/graphaug_<timestamp>/report.json`: per-question details (baseline + augmented top-k)

## Key knobs
- `graph_k`: number of neighbors per chunk in the chunk graph
- `hops`: how many graph hops to expand (1 is usually enough)
- `expand_per_node`: cap on how many neighbors are taken from each frontier node
- `beta`: blending weight for graph-proximity score (0 = baseline, 1 = graph-only)

## Recommended starting runs
- `graph_k=8, hops=1, expand_per_node=4, beta=0.25`
- `graph_k=16, hops=1, expand_per_node=8, beta=0.25`
