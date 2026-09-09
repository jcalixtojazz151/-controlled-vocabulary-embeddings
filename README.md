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

Project Structure

controlled-vocabulary-embeddings/
│
├── src/
│   ├── build_embeddings.py
│   ├── cpc_xml_parse.py
│   ├── mesh_xml_parse.py
│   └── embed_terms.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── ...

Generated source data, terms, and embeddings are intentionally excluded from Git and are stored locally during processing.

Installation

This project uses Python and uv for dependency management.

Create a virtual environment:

uv venv

Activate the virtual environment:

source .venv/bin/activate

Install the required packages:

uv pip install -r requirements.txt

Building Embeddings

The main entry point is:

python3 src/build_embeddings.py

The --vocab argument determines which vocabulary is processed.

Build CPC

python3 src/build_embeddings.py --vocab cpc

Build MeSH

python3 src/build_embeddings.py --vocab mesh

Build Both

python3 src/build_embeddings.py --vocab all

GPU / Apple Silicon

The embedding generation uses SentenceTransformers and supports specifying the computation device.

For Apple Silicon Macs, use the MPS backend:

python3 src/build_embeddings.py \
    --vocab all \
    --batch-size 32 \
    --device mps

For CUDA-enabled systems:

python3 src/build_embeddings.py \
    --vocab all \
    --batch-size 32 \
    --device cuda

For CPU:

python3 src/build_embeddings.py \
    --vocab all \
    --batch-size 32 \
    --device cpu

The batch size can be adjusted depending on available memory.

Command-Line Arguments

build_embeddings.py accepts the following arguments:

Argument	Description	Default
--vocab	Vocabulary to build: cpc, mesh, or all	Required
--batch-size	Embedding batch size	32
--device	SentenceTransformer device such as cpu, cuda, or mps	None

For example:

python3 src/build_embeddings.py \
    --vocab cpc \
    --batch-size 32 \
    --device mps

Processing Pipeline

For each vocabulary, the pipeline performs the following steps:

1. Download Source Data

The appropriate source archive is downloaded from the Hugging Face dataset repository:

jcalixto/compo_embeddings

CPC and MeSH use different source archives and release versions defined in build_embeddings.py.

2. Extract Source Data

The downloaded .tar.gz archive is extracted locally.

The extraction logic supports both:

* Archives containing a top-level directory
* Archives containing files directly at the archive root

3. Parse the Vocabulary

CPC XML files are parsed using:

src/cpc_xml_parse.py

MeSH XML files are parsed using:

src/mesh_xml_parse.py

The resulting vocabulary records are saved locally as JSON.

4. Generate Embeddings

Each vocabulary uses its corresponding SentenceTransformer model.

Vocabulary	Model
CPC	AI-Growth-Lab/PatentSBERTa
MeSH	NeuML/pubmedbert-base-embeddings

Embeddings are generated in batches and normalized before being saved.

5. Generate Metadata

Metadata is generated for every embedded term and saved alongside the embedding matrix.

6. Generate Manifest

A manifest.json file records information about:

* Vocabulary
* Version
* Source release
* Parser
* Embedding model
* Embedding dimension
* Number of terms
* Output artifact names

7. Upload Artifacts

The generated artifacts are uploaded back to the Hugging Face repository under the vocabulary/version directory.

The resulting structure is:

cpc/
└── 2026/
    ├── terms.json
    ├── embeddings.npy
    ├── metadata.json
    └── manifest.json
mesh/
└── 2026/
    ├── terms.json
    ├── embeddings.npy
    ├── metadata.json
    └── manifest.json

Output Files

For each vocabulary, the pipeline produces:

terms.json

The parsed controlled vocabulary terms and their associated information.

embeddings.npy

A NumPy matrix containing the normalized embeddings.

The number of rows corresponds to the number of terms, and the number of columns corresponds to the embedding dimension.

metadata.json

Metadata associated with each embedding. The ordering of metadata corresponds directly to the rows of embeddings.npy.

manifest.json

Describes the dataset version, source data, parser, embedding model, dimensions, and artifact structure.

Local Output

During processing, generated files are stored locally in:

data/
embeddings/
terms/

These directories are included in .gitignore because they contain generated and potentially very large files.

Hugging Face Authentication

The pipeline downloads and uploads files using the Hugging Face Hub.

If authentication is required, authenticate with:

huggingface-cli login

Do not commit authentication tokens or other credentials to this repository.

Reproducibility

The vocabulary version, source release, parser, embedding model, embedding dimension, and number of terms are recorded in each generated manifest.json.

This allows downstream applications to identify exactly which vocabulary and embedding configuration produced a given artifact.

License

See LICENSE for the terms under which this project is distributed.

:::