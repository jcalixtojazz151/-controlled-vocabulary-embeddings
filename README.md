# Controlled Vocabulary Embeddings

A pipeline for parsing controlled vocabulary sources and generating normalized semantic embeddings for use in downstream retrieval and analysis systems.

Currently supported vocabularies:

- **Cooperative Patent Classification (CPC)**
- **Medical Subject Headings (MeSH)**

The pipeline downloads source data from Hugging Face, extracts and parses the vocabulary files, generates embeddings using vocabulary-specific SentenceTransformer models, creates metadata and manifests, and uploads the resulting artifacts back to Hugging Face.

## Workflow

```text
Hugging Face
     │
     ├── CPC source archive
     │
     └── MeSH source archive
            │
            ▼
     Download source data
            │
            ▼
       Extract archives
            │
            ▼
      Parse vocabulary
       ┌────┴────┐
       ▼         ▼
      CPC       MeSH
       │         │
       └────┬────┘
            ▼
     Generate embeddings
            │
            ▼
   Generate metadata
            │
            ▼
      Generate manifest
            │
            ▼
    Upload to Hugging Face