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

## Validate

```
make validate
```

Runs the unit tests and the fixture evaluation.
