# Controlled Vocabulary Embeddings

Simple pipeline for parsing MeSH and CPC controlled vocabulary terms and generating embeddings.

## Workflow

1. Parse a MeSH XML source file with `mesh_xml_parse.py`
2. Parse CPC XML source files with `cpc_xml_parse.py`
3. Generate embeddings with `embed_terms.py`

## Installation

Install the required packages:

```bash
uv pip install -r requirements.txt