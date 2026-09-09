import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download, upload_file

from cpc_xml_parse import process_cpc_file
from mesh_xml_parse import process_mesh_file
from embed_terms import (
    build_cpc_embedding_text,
    build_mesh_embedding_text,
    build_cpc_metadata,
    build_mesh_metadata,
)

import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIGURATION
# =========================================================

HF_REPO_ID = "jcalixto/compo_embeddings"
HF_REPO_TYPE = "dataset"

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
TERMS_DIR = BASE_DIR / "terms"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

VOCAB_CONFIG = {
    "cpc": {
        "version": "2026",
        "source_archive": "cpc/source/FullCPCDefinitionXML202605.tar.gz",
        "local_archive": DATA_DIR / "cpc" / "FullCPCDefinitionXML202605.tar.gz",
        "local_source_dir": DATA_DIR / "cpc" / "FullCPCDefinitionXML202605",
        "terms_file": TERMS_DIR / "cpc_terms.json",
        "embedding_file": EMBEDDINGS_DIR / "cpc_embeddings.npy",
        "metadata_file": EMBEDDINGS_DIR / "cpc_metadata.json",
        "model": "AI-Growth-Lab/PatentSBERTa",
        "parser": "cpc_xml_parse.py",
        "source_name": "Cooperative Patent Classification",
        "source_release": "2026.05",
    },
    "mesh": {
        "version": "2026",
        "source_archive": "mesh/source/desc2026.xml.tar.gz",
        "local_archive": DATA_DIR / "mesh" / "desc2026.xml.tar.gz",
        "local_source_dir": DATA_DIR / "mesh" / "desc2026",
        "terms_file": TERMS_DIR / "mesh_terms.json",
        "embedding_file": EMBEDDINGS_DIR / "mesh_embeddings.npy",
        "metadata_file": EMBEDDINGS_DIR / "mesh_metadata.json",
        "model": "NeuML/pubmedbert-base-embeddings",
        "parser": "mesh_xml_parse.py",
        "source_name": "Medical Subject Headings",
        "source_release": "2026",
    },
}


# =========================================================
# HUGGING FACE
# =========================================================

def download_source_archive(config):
    """
    Download the source archive from Hugging Face.
    """

    print()
    print("========================================")
    print("Downloading source data")
    print("========================================")

    archive_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        filename=config["source_archive"],
        local_dir=DATA_DIR,
    )

    archive_path = Path(archive_path)

    # hf_hub_download may preserve the repository path.
    # Copy it to our predictable local archive location.
    config["local_archive"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if archive_path != config["local_archive"]:
        shutil.copy2(
            archive_path,
            config["local_archive"],
        )

    print(f"Downloaded: {config['source_archive']}")
    print(f"Saved to:   {config['local_archive']}")

    return config["local_archive"]


def upload_artifact(local_path, repo_path):
    """
    Upload one generated artifact to Hugging Face.
    """

    print(f"Uploading {local_path} → {repo_path}")

    upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
    )


# =========================================================
# EXTRACTION
# =========================================================

def extract_archive(archive_path, extract_dir):
    """
    Extract a tar.gz archive.

    Handles both:
    1. Archives containing a top-level directory, e.g.
       FullCPCDefinitionXML202605/cpc-definition-G06E.xml

    2. Archives containing a single file, e.g.
       desc2026.xml
    """

    print()
    print("========================================")
    print("Extracting source data")
    print("========================================")

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"Source directory already exists: {extract_dir}")
        return

    extract_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        archive_path,
        mode="r:gz",
    ) as tar:

        members = tar.getmembers()

        # Get the top-level paths in the archive
        top_level_names = {
            member.name.split("/")[0]
            for member in members
            if member.name
        }

        # If the archive contains a single top-level directory,
        # extract into its parent so that directory is preserved.
        if len(top_level_names) == 1:
            top_level_name = next(iter(top_level_names))

            has_top_level_directory = any(
                member.name.rstrip("/") == top_level_name
                and member.isdir()
                for member in members
            )

            if has_top_level_directory:
                extraction_path = extract_dir.parent
            else:
                extraction_path = extract_dir

        else:
            extraction_path = extract_dir

        tar.extractall(
            path=extraction_path
        )

    print(f"Extracted to: {extract_dir}")


# =========================================================
# CPC PARSING
# =========================================================

def parse_cpc(config):
    """
    Parse CPC XML files into structured records.
    """

    print()
    print("========================================")
    print("Parsing CPC")
    print("========================================")

    source_dir = config["local_source_dir"]

    if not source_dir.exists():
        raise RuntimeError(
            f"CPC source directory does not exist: {source_dir}"
        )

    # Search recursively because the CPC archive may contain
    # one or more levels of nested directories.
    xml_files = sorted(
        f
        for f in source_dir.rglob("cpc-definition-*.xml")
        if f.is_file()
    )

    if not xml_files:
        raise RuntimeError(
            f"No CPC XML files found in {source_dir}"
        )

    all_records = []

    for xml_file in xml_files:

        print(f"Processing: {xml_file}")

        records = process_cpc_file(
            str(xml_file)
        )

        all_records.extend(records)

    config["terms_file"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with config["terms_file"].open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_records,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Processed {len(xml_files)} CPC XML files.")
    print(f"Parsed {len(all_records)} CPC terms.")
    print(f"Saved to: {config['terms_file']}")

    return all_records


# =========================================================
# MESH PARSING
# =========================================================

def parse_mesh(config):
    """
    Parse MeSH XML into structured records.
    """

    print()
    print("========================================")
    print("Parsing MeSH")
    print("========================================")

    source_dir = config["local_source_dir"]

    if not source_dir.exists():
        raise RuntimeError(
            f"MeSH source directory does not exist: {source_dir}"
        )

    xml_files = list(
        source_dir.rglob("*.xml")
    )

    if not xml_files:
        raise RuntimeError(
            f"No MeSH XML files found in {source_dir}"
        )

    # desc2026.xml is normally the actual MeSH descriptor file.
    xml_path = next(
        (
            path
            for path in xml_files
            if path.name == "desc2026.xml"
        ),
        None,
    )

    if xml_path is None:
        if len(xml_files) == 1:
            xml_path = xml_files[0]
        else:
            raise RuntimeError(
                "Could not identify desc2026.xml in the "
                f"extracted MeSH source: {source_dir}"
            )

    print(f"Processing: {xml_path}")

    records = process_mesh_file(
        str(xml_path)
    )

    config["terms_file"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with config["terms_file"].open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Parsed {len(records)} MeSH terms.")
    print(f"Saved to: {config['terms_file']}")

    return records


# =========================================================
# EMBEDDING
# =========================================================

def build_embedding_text(term_type, record):
    """
    Build the embedding representation using the existing
    embedding-text logic.
    """

    if term_type == "cpc":
        return build_cpc_embedding_text(record)

    if term_type == "mesh":
        return build_mesh_embedding_text(record)

    raise ValueError(
        f"Unsupported vocabulary: {term_type}"
    )


def build_metadata(term_type, record):
    """
    Build metadata using the existing metadata logic.
    """

    if term_type == "cpc":
        return build_cpc_metadata(record)

    if term_type == "mesh":
        return build_mesh_metadata(record)

    raise ValueError(
        f"Unsupported vocabulary: {term_type}"
    )


def generate_embeddings(
    term_type,
    config,
    records,
    batch_size=32,
    device=None,
):
    """
    Generate embeddings and metadata.
    """

    print()
    print("========================================")
    print(f"Generating {term_type.upper()} embeddings")
    print("========================================")

    texts = []
    metadata = []

    for record in records:

        text = build_embedding_text(
            term_type,
            record,
        ).strip()

        if not text:
            continue

        texts.append(text)

        metadata.append(
            build_metadata(
                term_type,
                record,
            )
        )

    print(f"Prepared {len(texts)} embedding texts.")

    if not texts:
        raise RuntimeError(
            f"No valid embedding text found for {term_type}."
        )

    if len(texts) != len(metadata):
        raise RuntimeError(
            f"Text/metadata mismatch: "
            f"{len(texts)} vs {len(metadata)}"
        )

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print(f"Loading model: {config['model']}")
    print(f"Device: {device}")

    model = SentenceTransformer(
        config["model"],
        device=device,
    )

    print(f"Model device: {model.device}")

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    print(
        f"Generating {len(texts)} embeddings..."
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Expected 2D embeddings, got "
            f"{embeddings.shape}"
        )

    if embeddings.shape[0] != len(metadata):
        raise RuntimeError(
            f"Embedding/metadata mismatch: "
            f"{embeddings.shape[0]} vs {len(metadata)}"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embedding matrix contains NaN or infinite values."
        )

    print(f"Embedding shape: {embeddings.shape}")

    # -----------------------------------------------------
    # Save embeddings
    # -----------------------------------------------------

    config["embedding_file"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        config["embedding_file"],
        embeddings,
    )

    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    with config["metadata_file"].open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved embeddings: {config['embedding_file']}")
    print(f"Saved metadata:   {config['metadata_file']}")

    return embeddings, metadata


# =========================================================
# MANIFEST
# =========================================================

def build_manifest(
    term_type,
    config,
    num_terms,
    embedding_dimension,
):
    """
    Build the Hugging Face manifest.
    """

    manifest = {
        "artifact_type": "term_embeddings",
        "term_type": term_type,
        "version": config["version"],
        "source": {
            "name": config["source_name"],
            "release": config["source_release"],
            "archive": (
                f"source/"
                f"{Path(config['source_archive']).name}"
            ),
        },
        "processing": {
            "parser": config["parser"],
            "parsed_terms": "terms.json",
        },
        "embedding": {
            "model": config["model"],
            "dimension": embedding_dimension,
            "normalized": True,
        },
        "data": {
            "num_terms": num_terms,
            "terms_file": "terms.json",
            "embedding_file": "embeddings.npy",
            "metadata_file": "metadata.json",
        },
    }

    return manifest


def save_manifest(
    term_type,
    config,
    num_terms,
    embedding_dimension,
):
    """
    Save manifest.json locally.
    """

    manifest = build_manifest(
        term_type=term_type,
        config=config,
        num_terms=num_terms,
        embedding_dimension=embedding_dimension,
    )

    manifest_path = (
        EMBEDDINGS_DIR
        / f"{term_type}_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved manifest: {manifest_path}")

    return manifest_path


# =========================================================
# BUILD ONE VOCABULARY
# =========================================================

def build_vocabulary(
    term_type,
    batch_size=32,
    device=None,
):
    """
    Build one complete vocabulary.
    """

    config = VOCAB_CONFIG[term_type]

    print()
    print("########################################")
    print(f"BUILDING {term_type.upper()}")
    print("########################################")

    # -----------------------------------------------------
    # 1. Download source
    # -----------------------------------------------------

    archive_path = download_source_archive(
        config
    )

    # -----------------------------------------------------
    # 2. Extract source
    # -----------------------------------------------------

    extract_archive(
        archive_path,
        config["local_source_dir"],
    )

    # -----------------------------------------------------
    # 3. Parse source
    # -----------------------------------------------------

    if term_type == "cpc":
        records = parse_cpc(config)

    elif term_type == "mesh":
        records = parse_mesh(config)

    else:
        raise ValueError(
            f"Unsupported vocabulary: {term_type}"
        )

    # -----------------------------------------------------
    # 4. Generate embeddings
    # -----------------------------------------------------

    embeddings, metadata = generate_embeddings(
        term_type=term_type,
        config=config,
        records=records,
        batch_size=batch_size,
        device=device,
    )

    # -----------------------------------------------------
    # 5. Generate manifest
    # -----------------------------------------------------

    manifest_path = save_manifest(
        term_type=term_type,
        config=config,
        num_terms=len(metadata),
        embedding_dimension=embeddings.shape[1],
    )

    # -----------------------------------------------------
    # 6. Upload generated artifacts
    # -----------------------------------------------------

    print()
    print("========================================")
    print("Uploading artifacts")
    print("========================================")

    hf_prefix = (
        f"{term_type}/{config['version']}"
    )

    upload_artifact(
        config["terms_file"],
        f"{hf_prefix}/terms.json",
    )

    upload_artifact(
        config["embedding_file"],
        f"{hf_prefix}/embeddings.npy",
    )

    upload_artifact(
        config["metadata_file"],
        f"{hf_prefix}/metadata.json",
    )

    upload_artifact(
        manifest_path,
        f"{hf_prefix}/manifest.json",
    )

    print()
    print("========================================")
    print(f"{term_type.upper()} BUILD COMPLETE")
    print("========================================")
    print(f"Terms:       {len(metadata)}")
    print(f"Dimensions:  {embeddings.shape[1]}")
    print(f"Model:       {config['model']}")
    print(f"HF path:     {hf_prefix}/")
    print("========================================")


# =========================================================
# ARGUMENTS
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Download controlled vocabulary source data, "
            "generate embeddings, metadata, and manifests, "
            "and upload the results to Hugging Face."
        )
    )

    parser.add_argument(
        "--vocab",
        choices=["cpc", "mesh", "all"],
        required=True,
        help="Vocabulary to build.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help=(
            "SentenceTransformer device "
            "(e.g. cpu, cuda, mps)."
        ),
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================

def main():

    args = parse_args()

    if args.vocab == "all":

        build_vocabulary(
            "cpc",
            batch_size=args.batch_size,
            device=args.device,
        )

        build_vocabulary(
            "mesh",
            batch_size=args.batch_size,
            device=args.device,
        )

    else:

        build_vocabulary(
            args.vocab,
            batch_size=args.batch_size,
            device=args.device,
        )


if __name__ == "__main__":
    main()