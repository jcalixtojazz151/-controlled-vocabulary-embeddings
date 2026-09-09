import re
import os
import json
import xml.etree.ElementTree as ET

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Normalize CPC text.
    """

    if not text:
        return ""
    
    # Remove braces
    text = text.replace("{", "").replace("}", "")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def normalize_cpc_code(code):
    """
    Normalize CPC code format.
    """

    if not code:
        return ""
    
    code = code.strip()

    # Remove internal spaces
    code = re.sub(r"\s+", "", code)

    return code.upper()

def paragraph_text(element):
    """
    Extract all text from an XML element
    """

    if element is None:
        return ""
    
    text = " ".join(
        t.strip()
        for t in element.itertext()
        if t and t.strip()
    )

    return clean_text(text)


# =========================================================
# CPC CODE
# =========================================================

def extract_cpc_code(definition):

    symbol = definition.find("classification-symbol")

    if symbol is None or not symbol.text:
        return ""
    
    return normalize_cpc_code(symbol.text)

# =========================================================
# TITLE
# =========================================================

def extract_title(definition):

    title_node = definition.find("definition-title")

    return paragraph_text(title_node)

# =========================================================
# DEFINITION TEXT
# =========================================================

def extract_definition_text(definition):

    boilerplate_phrases = {
        "This place covers:"
    }
    
    statement = definition.find("definition-statement")

    if statement is None:
        return ""
    
    paragraphs = []

    for p in statement.findall(".//paragraph-text"):

        text = paragraph_text(p)

        if text in boilerplate_phrases:
            continue

        if text:
            paragraphs.append(text)

    return " ".join(paragraphs)

# =========================================================
# GLOSSARY
# =========================================================

def extract_glossary(definition):

    glossary = {}

    rows = definition.findall(".//glossary-of-terms//table-row")

    for row in rows:

        cols = row.findall("./table-column")

        if len(cols) < 2:
            continue

        term = paragraph_text(cols[0])
        meaning = paragraph_text(cols[1])

        if term and meaning:
            glossary[term] = meaning

    return glossary

# =========================================================
# SYNONYMS
# =========================================================

def extract_synonyms_text(definition):

    boilerplate_phrases = {
        "In patent documents, the following words/expressions are often used as synonyms:", 
        "In patent documents, the following abbreviations are often used:", 
        "The following terms or expressions are used with the meaning indicated:"
    }

    synonym_texts = []

    sections = definition.findall(".//synonyms-keywords")

    for section in sections:

        for p in section.findall(".//paragraph-text"):

            text = paragraph_text(p)

            if not text:
                continue

            if text in boilerplate_phrases:
                continue

            synonym_texts.append(text)

    synonyms = sorted(set(synonym_texts))

    return " | ".join(synonyms)

# =========================================================
# CPC HIERARCHY
# =========================================================

# def get_parent_codes(cpc_code):
#     """
#     Compute CPC parents using structural rules.

#     Example:
#     G06E3/001 -> ["G06E", "G06E3/00"]
#     """

#     if not cpc_code:
#         return []
    
#     parents = []

#     match = re.match(r"([A-Z])(\d{2})([A-Z])", cpc_code)

#     if not match:
#         return []

#     section = match.group(1)
#     class_code = section + match.group(2)
#     subclass = class_code + match.group(3)

#     parents.append(subclass)

#     if "/" in cpc_code:

#         left, right = cpc_code.split("/")

#         if right != "00":
#             main_group = (left + "/00")
#             parents.append(main_group)

#     return sorted(set(parents), key=len)

def get_parent_codes(cpc_code, all_codes):
    """
    Find all ancestor CPC codes that exist in dataset.
    """

    parents = []

    if not cpc_code:
        return []

    subclass = cpc_code[:4]

    if (subclass != cpc_code and subclass in all_codes):
        
        parents.append(subclass)

    if "/" in cpc_code:

        left, right = (cpc_code.split("/"))

        main_group = (left + "/00")

        if (main_group != cpc_code and main_group in all_codes):

            parents.append(main_group)

        for i in range(2, len(right)):

            subgroup = (left + "/" + right[:i])

            if (subgroup in all_codes and subgroup != cpc_code):
                parents.append(subgroup)

    return sorted(set(parents), key=len)

def get_cpc_metadata(cpc_code):

    metadata = {
        "section": "", 
        "class": "", 
        "subclass": "", 
        "main_group": "", 
        "level": 0
    }

    if not cpc_code:
        return  metadata
    
    metadata["section"] = cpc_code[0]

    metadata["class"] = cpc_code[:3]

    metadata["subclass"] = cpc_code[:4]

    if "/" in cpc_code:

        left, right = cpc_code.split("/")

        metadata["main_group"] = (left + "/00")

        metadata["level"] = 3

    else:

        metadata["level"] = 1

    return metadata

# =========================================================
# CPC EMBEDDING TEXT
# =========================================================

def build_embedding_text(title, parent_titles_text, definition_text, glossary, synonyms_text):

    parts = []

    if title:
        parts.append(f"Technology: {title}")

    if parent_titles_text:
        parts.append(f"Parent categories: {parent_titles_text}")

    if definition_text:
        parts.append(f"Definition: {definition_text}")

    if glossary:

        glossary_text = " ".join(
            f"{term}: {meaning}"
            for term, meaning in glossary.items()
        )

        parts.append(f"Glossary: {glossary_text}")

    if synonyms_text:
        parts.append(f"Synonyms: {synonyms_text}")

    return " | ".join(parts)

# def build_embedding_text(hierarchical_title, title, definition_text, glossary, synonyms_text):

#     parts = []

#     if hierarchical_title:
#         parts.append(hierarchical_title)
#     else:
#         parts.append(title)

#     if definition_text:
#         parts.append(definition_text)

#     if glossary:

#         glossary_text = " ".join(
#             f"{term}: {meaning}"
#             for term, meaning in glossary.items()
#         )

#         parts.append(glossary_text)

#     if synonyms_text:
#         parts.append(synonyms_text)

#     return " | ".join(parts)

# =========================================================
# PROCESS SINGLE CPC XML FILE
# =========================================================

def process_cpc_file(xml_path):

    records = []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    definition_items = root.findall(".//definition-item")

    # First pass: build lookup tables and collect all CPC codes
    
    code_to_title = {}
    all_codes = set()
    
    for definition in definition_items:

        cpc_code = extract_cpc_code(definition)

        if not cpc_code:
            continue

        title = extract_title(definition)
        cpc_code = normalize_cpc_code(cpc_code)

        all_codes.add(cpc_code)
        code_to_title[cpc_code] = title

    # Second pass: Build final structured records

    for definition in definition_items:

        cpc_code = extract_cpc_code(definition)

        if not cpc_code:
            continue

        cpc_code = normalize_cpc_code(cpc_code)

        title = extract_title(definition)
        definition_text = extract_definition_text(definition)
        glossary = extract_glossary(definition)
        synonyms_text = extract_synonyms_text(definition)

        # Hierarchy: derive parents from known CPC codes

        parents = get_parent_codes(cpc_code=cpc_code, all_codes=all_codes)

        parent_titles = [code_to_title.get(p, "") for p in parents]

        # Remove empty parent titles
        parent_titles = [t for t in parent_titles if t]

        # Level (MeSH-like depth)
        level = len(parents) + 1

        # Hierarchical title
        # hierarchical_title = " | ".join(parent_titles + [title])
        parent_titles_text = " | ".join(parent_titles[-2:])

        embedding_text = build_embedding_text(title=title, parent_titles_text=parent_titles_text, definition_text=definition_text, glossary=glossary, synonyms_text=synonyms_text)

        # Final record
        record = {
            "cpc_code": cpc_code, 
            "title": title, 
            "definition_text": definition_text, 
            "glossary": glossary, 
            "synonyms_text": synonyms_text, 

            # hierarchy
            "parents": parents, 
            "parent_titles": parent_titles, 
            "level": level, 

            #embedding
            "embedding_text": embedding_text
        }

        records.append(record)

    return records

# =========================================================
# MAIN
# =========================================================

def main():

    cpc_folder = "data/cpc/FullCPCDefinitionXML202605"

    xml_files = [
        f for f in os.listdir(cpc_folder)
        if f.startswith("cpc-definition-") and f.endswith(".xml")
    ]

    all_records = []

    for xml_file in xml_files:

        xml_path = os.path.join(cpc_folder, xml_file)

        print(f"Processing: {xml_file}")

        records = process_cpc_file(xml_path)

        all_records.extend(records)

    terms_dir = "terms"
    cpc_terms_json_path = "cpc_terms.json"

    # Create terms folder if it doesn't exist
    os.makedirs(terms_dir, exist_ok=True)

    
    output_path = os.path.join(terms_dir, cpc_terms_json_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"Process {len(xml_files)} CPC xml files.")
    print(f"Saved {len(all_records)} CPC records to {output_path}")

if __name__ == "__main__":
    main()

