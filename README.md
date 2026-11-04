# rag-bm25-proximity-rerank

BM25 scores a document by which query terms it contains, not by where they
sit. "connection pool exhausted" matches a page that mentions the three words
in three unrelated paragraphs as well as one that says the phrase outright.

This repo adds a second pass: take the BM25 top-N, measure how tightly the
query terms cluster in each document, and add a bounded bonus for tight
clusters. Pure stdlib Python, no model, no network.

## How it works

1. `proxrank.bm25` indexes token positions and returns a BM25 ranking.
2. `proxrank.proximity` finds the smallest token window that holds every
   query term that occurs in the document.
3. `proxrank.rerank` adds `weight * coverage / span` to the BM25 score of the
   top candidates, where `coverage` is the share of query terms present and
   `span` is the window length. A perfect phrase has `span == terms`.

## Usage

```
python3 -m proxrank "connection pool exhausted" --corpus fixtures/docs.json
```

## Results

`make eval` runs 8 labelled queries over 14 fixture documents. Six queries have
a decoy document that repeats every query term far apart; two are easy
controls. These are fixture numbers, not a benchmark.

| run | recall@3 | MRR |
| --- | --- | --- |
| plain BM25 | 1.000 | 0.625 |
| proximity weight 0.5 | 1.000 | 0.812 |
| proximity weight 1 | 1.000 | 1.000 |
| proximity weight 2 | 1.000 | 1.000 |

The fixture is small and built to contain the failure mode, so the gain shows
that the rerank works as intended, not how much it helps on real text.

## Caveats

- Only the BM25 top `--candidates` (default 20) are reranked.
- Order does not matter: "pool connection" and "connection pool" both count as
  a tight cluster.
- Terms the document lacks are ignored in the span but lower `coverage`.

## Validate

```
make validate
```

Runs the unit tests and the fixture evaluation.
