import os
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# LOAD JSON
# =========================================================

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# METADATA
# =========================================================

def build_mesh_metadata(record):
    return {
        "mesh_id": record.get("mesh_id"),
        "name": record.get("mesh_term_name"),
        "scope_note": record.get("scope_note"),
        "tree_numbers": record.get("tree_numbers", []),
        "concept_terms": record.get("concept_terms", []),
        "parent_titles": record.get("parent_titles", []),
    }


def build_cpc_metadata(record):
    return {
        "cpc_code": record.get("cpc_code"),
        "name": record.get("title"),
        "definition": record.get("definition_text"),
        "glossary": dict(record.get("glossary", {})),
        "synonyms_text": record.get("synonyms_text"),
        "parents": record.get("parents"),
        "parent_titles": record.get("parent_titles"),
        "level": record.get("level"),
    }


# =========================================================
# EMBEDDING TEXT
# =========================================================

def build_cpc_embedding_text(term):
    """
    Build CPC semantic representation.

    Structure:
        CPC Concept: parent titles | concept title
        Definition: definition
        Glossary: term: meaning; term: meaning
    """

    title = term.get("title", "").strip()

    parent_titles = [
        parent.strip()
        for parent in term.get("parent_titles", [])
        if parent and parent.strip()
    ]

    definition = term.get("definition_text", "").strip()

    glossary = term.get("glossary", {})

    parts = []

    # -----------------------------------------------------
    # Concept hierarchy
    # -----------------------------------------------------

    if title and parent_titles:
        concept_text = " | ".join(parent_titles + [title])
        parts.append(f"CPC Concept: {concept_text}")

    elif title:
        parts.append(f"CPC Concept: {title}")

    # -----------------------------------------------------
    # Definition
    # -----------------------------------------------------

    if definition:
        parts.append(f"Definition: {definition}")

    # -----------------------------------------------------
    # Glossary
    # -----------------------------------------------------

    if glossary:

        glossary_text = "; ".join(
            f"{term}: {meaning}"
            for term, meaning in glossary.items()
            if term and meaning
        )

        if glossary_text:
            parts.append(f"Glossary: {glossary_text}")

    return " | ".join(parts)


def build_mesh_embedding_text(term):
    """
    Build MeSH semantic representation.

    Structure:
        MeSH Concept: parent titles | concept title
        Scope Note: scope note
        Concept Terms: term | term | term
    """

    name = term.get("mesh_term_name", "").strip()

    parent_titles = [
        parent.strip()
        for parent in term.get("parent_titles", [])
        if parent and parent.strip()
    ]

    scope_note = term.get("scope_note", "").strip()

    concept_terms = [
        term.strip()
        for term in term.get("concept_terms", [])
        if term and term.strip()
    ]

    parts = []

    # -----------------------------------------------------
    # Concept hierarchy
    # -----------------------------------------------------

    if name and parent_titles:
        concept_text = " | ".join(parent_titles + [name])
        parts.append(f"MeSH Concept: {concept_text}")

    elif name:
        parts.append(f"MeSH Concept: {name}")

    # -----------------------------------------------------
    # Scope note
    # -----------------------------------------------------

    if scope_note:
        parts.append(f"Scope Note: {scope_note}")

    # -----------------------------------------------------
    # Concept terms
    # -----------------------------------------------------

    if concept_terms:
        parts.append(
            f"Concept Terms: {' | '.join(concept_terms)}"
        )

    return " | ".join(parts)

def build_embedding_text(term_type, record):

    if term_type == "cpc":
        return build_cpc_embedding_text(record)

    if term_type == "mesh":
        return build_mesh_embedding_text(record)

    raise ValueError(
        f"Term type {term_type} is not allowed. "
        "Choose from: cpc, mesh."
    )


# =========================================================
# EMBEDDING
# =========================================================

def embed_records(
    term_type,
    input_json_path,
    output_embedding_path,
    output_metadata_path,
    bi_encoder_model_name,
    batch_size=32,
    normalize_embeddings=True,
    device=None,
):

    # -----------------------------------------------------
    # 1. Load records
    # -----------------------------------------------------

    records = load_json(input_json_path)

    print(f"Loaded {len(records)} raw {term_type} records.")


    # -----------------------------------------------------
    # 2. Build embedding texts + metadata
    # -----------------------------------------------------

    texts = []
    metadata = []

    for record in records:

        text = build_embedding_text(
            term_type=term_type,
            record=record,
        ).strip()

        if not text:
            continue

        texts.append(text)

        if term_type == "mesh":
            metadata.append(
                build_mesh_metadata(record)
            )

        elif term_type == "cpc":
            metadata.append(
                build_cpc_metadata(record)
            )

    print(f"Prepared {len(texts)} {term_type} texts for embedding.")

    if not texts:
        raise RuntimeError(
            f"No valid embedding text found for {term_type}."
        )

    if len(texts) != len(metadata):
        raise RuntimeError(
            f"Text/metadata mismatch: "
            f"{len(texts)} texts vs {len(metadata)} metadata records."
        )


    # -----------------------------------------------------
    # 3. Load model
    # -----------------------------------------------------

    print(f"Loading {bi_encoder_model_name}")
    print(f"Using device: {device}")

    model = SentenceTransformer(
        bi_encoder_model_name,
        device=device,
    )

    print(f"Model device: {model.device}")


    # -----------------------------------------------------
    # 4. Generate embeddings
    # -----------------------------------------------------

    print(
        f"Generating {len(texts)} {term_type} embeddings "
        f"with batch size {batch_size}..."
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )


    # -----------------------------------------------------
    # 5. Validate
    # -----------------------------------------------------

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Expected 2D embeddings, got {embeddings.shape}."
        )

    if embeddings.shape[0] != len(texts):
        raise RuntimeError(
            f"Generated {embeddings.shape[0]} embeddings "
            f"for {len(texts)} texts."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embedding matrix contains NaN or infinite values."
        )

    print(f"Embedding shape: {embeddings.shape}")


    # -----------------------------------------------------
    # 6. Save embeddings
    # -----------------------------------------------------

    output_embedding_path = Path(output_embedding_path)
    output_embedding_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_embedding_path,
        embeddings,
    )


    # -----------------------------------------------------
    # 7. Save metadata
    # -----------------------------------------------------

    output_metadata_path = Path(output_metadata_path)
    output_metadata_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # -----------------------------------------------------
    # 8. Final report
    # -----------------------------------------------------

    print()
    print("========================================")
    print("Embedding generation complete")
    print("========================================")
    print(f"Vocabulary:       {term_type}")
    print(f"Model:            {bi_encoder_model_name}")
    print(f"Device:           {model.device}")
    print(f"Terms embedded:   {len(texts)}")
    print(f"Embedding shape:  {embeddings.shape}")
    print(f"Embeddings:       {output_embedding_path}")
    print(f"Metadata:         {output_metadata_path}")
    print("========================================")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    embeddings_dir = "embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)


    # =====================================================
    # CPC
    # =====================================================

    cpc_json_path = "terms/cpc_terms.json"

    cpc_embedding_path = os.path.join(
        embeddings_dir,
        "cpc_embeddings.npy",
    )

    cpc_metadata_path = os.path.join(
        embeddings_dir,
        "cpc_metadata.json",
    )

    embed_records(
        term_type="cpc",
        input_json_path=cpc_json_path,
        output_embedding_path=cpc_embedding_path,
        output_metadata_path=cpc_metadata_path,
        bi_encoder_model_name="AI-Growth-Lab/PatentSBERTa",
        batch_size=32,
        normalize_embeddings=True,
        device=None,
    )


    # =====================================================
    # MeSH
    # =====================================================

    mesh_json_path = "terms/mesh_terms.json"

    mesh_embedding_path = os.path.join(
        embeddings_dir,
        "mesh_embeddings.npy",
    )

    mesh_metadata_path = os.path.join(
        embeddings_dir,
        "mesh_metadata.json",
    )

    embed_records(
        term_type="mesh",
        input_json_path=mesh_json_path,
        output_embedding_path=mesh_embedding_path,
        output_metadata_path=mesh_metadata_path,
        bi_encoder_model_name="NeuML/pubmedbert-base-embeddings",
        batch_size=32,
        normalize_embeddings=True,
        device=None,
    )