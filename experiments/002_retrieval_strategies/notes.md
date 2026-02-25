# Retrieval strategy experiments

Run everything (dense, BM25, hybrid, parent-child, rerank, rewriting):

./experiments/002_retrieval_strategies/run_matrix.ps1

./experiments/002_retrieval_strategies/run_matrix.ps1 -OnlyHybridAndRerank

./experiments/002_retrieval_strategies/run_matrix.ps1 -IncludeOverlap300100

./experiments/002_retrieval_strategies/run_matrix.ps1 -IncludeBioEmbedding

Outputs:
- `results/runs/retrieval_<timestamp>/...`
- `results/summary/leaderboard_retrieval.csv`
