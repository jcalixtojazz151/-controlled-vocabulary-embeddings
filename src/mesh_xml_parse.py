import os
import re
import json
import xml.etree.ElementTree as ET


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Normalize MeSH text.
    """

    if not text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def paragraph_text(element):
    """
    Extract all text from an XML element.
    """

    if element is None:
        return ""
    
    text = " ".join(
        t.strip()
        for t in element.itertext()
        if t and t.strip()
    )

    return clean_text(text)

def extract_text(element, path):
    """
    Extract text from first matching XML path.
    """

    node = element.find(path)
    
    if node is None:
        return ""
    
    return paragraph_text(node)

def extract_list(element, path):
    """
    Extract list of text values from XML path.
    """

    values = []

    nodes = element.findall(path)

    for node in nodes:
        
        text = paragraph_text(node)

        if text:
            values.append(text)

    return sorted(set(values))


# =========================================================
# MESH ID
# =========================================================

def extract_mesh_id(record):

    return extract_text(record, "DescriptorUI")

# =========================================================
# MESH TERM NAME
# =========================================================

def extract_mesh_term_name(record):

    return extract_text(record, "DescriptorName/String")

# =========================================================
# SCOPE NOTE
# =========================================================

def extract_scope_note(record):

    return extract_text(record, "ConceptList/Concept/ScopeNote")

# =========================================================
# TREE NUMBERS
# =========================================================

def extract_tree_numbers(record):

    return extract_list(record, "TreeNumberList/TreeNumber")

def extract_parent_tree_numbers(tree_numbers):
    """
    Return immediate parent tree numbers for a list of MeSH tree numbers.
    """

    parent_tree_numbers = set()

    for tree_number in tree_numbers:

        if "." not in tree_number:
            continue

        parent = tree_number.rsplit(".", 1)[0]

        parent_tree_numbers.add(parent)

    return sorted(parent_tree_numbers)

# =========================================================
# CONCEPT TERMS
# =========================================================

def extract_concept_terms(record):

    return extract_list(record, "ConceptList/Concept/TermList/Term/String")

# =========================================================
# PROCESS SINGLE MESH RECORD
# =========================================================

def process_mesh_record(record):

    mesh_id = extract_mesh_id(record)

    if not mesh_id:
        return None
    
    mesh_term_name = extract_mesh_term_name(record)

    scope_note = extract_scope_note(record)

    tree_numbers = extract_tree_numbers(record)

    concept_terms = extract_concept_terms(record)

    # Final record
    return {
        "mesh_id": mesh_id,
        "mesh_term_name": mesh_term_name,
        "scope_note": scope_note,
        "tree_numbers": tree_numbers,
        "concept_terms": concept_terms,
        "parent_titles": [],
    }

# =========================================================
# PROCESS MESH XML FILE
# =========================================================

def process_mesh_file(xml_path):

    records = []

    context = ET.iterparse(xml_path, events=("end",))

    for event, elem in context:

        if elem.tag == "DescriptorRecord":

            mesh_record = process_mesh_record(elem)

            if mesh_record:
                records.append(mesh_record)

            elem.clear()

    # -----------------------------------------------------
    # Build tree number → descriptor name lookup
    # -----------------------------------------------------

    tree_to_name = {}

    for record in records:

        name = record["mesh_term_name"]

        for tree_number in record["tree_numbers"]:

            tree_to_name[tree_number] = name

    # -----------------------------------------------------
    # Resolve parent titles
    # -----------------------------------------------------

    for record in records:

        parent_tree_numbers = extract_parent_tree_numbers(
            record["tree_numbers"]
        )

        parent_titles = []

        for parent_tree_number in parent_tree_numbers:

            parent_title = tree_to_name.get(parent_tree_number)

            if parent_title:
                parent_titles.append(parent_title)

        record["parent_titles"] = sorted(set(parent_titles))

    return records

# =========================================================
# MAIN
# =========================================================

def main():

    # Define paths
    mesh_xml_path = "data/mesh/desc2026.xml"
    terms_dir = "terms"
    mesh_terms_json_path = "mesh_terms.json"

    os.makedirs(terms_dir, exist_ok=True)

    output_path = os.path.join(terms_dir, mesh_terms_json_path)

    print(f"Processing: {mesh_xml_path}")

    records = process_mesh_file(mesh_xml_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} MeSH records to {output_path}")

if __name__ == "__main__":
    main()