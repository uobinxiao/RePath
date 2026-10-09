#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a task-agnostic digital pathology prompt bank for CONCH-style text encoding.

This script only generates prompts. It does not compute embeddings.

Outputs:
  1. prompts.jsonl
     One prompt per line with concept/category/template metadata.

  2. prompts.csv
     Same content in CSV format.

  3. prompts.txt
     Flat prompt list, one prompt per line.

Example:
  python generate_pathology_prompts.py --out_dir ./prompt_bank
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Any


# ---------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------

TEMPLATES: List[str] = [
    "a hematoxylin and eosin histopathology image showing {concept}",
    "an H&E stained histology image showing {concept}",
    "a pathology image of tissue with {concept}",
    "a microscopic histopathology image demonstrating {concept}",
    "a digital pathology image containing {concept}",
]


# ---------------------------------------------------------------------
# Positive concepts: general pathology morphology primitives
# ---------------------------------------------------------------------

CONCEPTS: Dict[str, Dict[str, List[str]]] = {
    "positive": {
        # Basic tissue compartments and microenvironment
        "tissue_compartment": [
            "epithelial tissue",
            "tumor epithelium",
            "benign epithelial tissue",
            "normal glandular epithelium",
            "stromal tissue",
            "fibrous stroma",
            "desmoplastic stroma",
            "loose connective tissue",
            "smooth muscle tissue",
            "adipose tissue",
            "lymphoid tissue",
            "immune cell infiltrate",
            "lymphocyte-rich inflammation",
            "plasma cell-rich inflammation",
            "blood vessels",
            "vascular structures",
            "blood-rich tissue",
            "hemorrhagic tissue",
            "epithelial-stromal interface",
            "tumor-stroma interface",
            "tissue boundary",
        ],

        # Architecture-level concepts, useful for cross-tissue transfer
        "architecture": [
            "glandular architecture",
            "ductal structures",
            "acinar structures",
            "crypt architecture",
            "lumen formation",
            "well-formed glands",
            "irregular glands",
            "crowded glands",
            "fused glands",
            "small glandular structures",
            "large glandular structures",
            "back-to-back glands",
            "cribriform-like architecture",
            "papillary architecture",
            "tubular architecture",
            "trabecular architecture",
            "nested epithelial growth",
            "solid sheet-like growth",
            "solid tumor nests",
            "epithelial folds",
            "serrated epithelial architecture",
            "villous architecture",
            "invasive tumor front",
            "infiltrative growth pattern",
            "expansile growth pattern",
            "tissue-level architecture",
            "large field of view tissue structure",
            "gland-level architecture",
        ],

        # Tumor morphology and common pathology states
        "pathology_state": [
            "tumor nests",
            "solid tumor growth",
            "invasive carcinoma",
            "carcinoma in situ",
            "benign tumor tissue",
            "normal tissue architecture",
            "atypical epithelial cells",
            "dysplastic epithelium",
            "high cellularity tumor",
            "low cellularity tissue",
            "mucin-rich tissue",
            "extracellular mucin",
            "intracellular mucin",
            "necrosis",
            "tumor necrosis",
            "comedonecrosis-like morphology",
            "fibrosis",
            "scar-like fibrosis",
            "inflammation",
            "chronic inflammation",
            "acute inflammation",
            "granulation tissue",
            "edema",
            "calcification",
            "keratinization",
            "squamous differentiation",
            "clear cell morphology",
            "pigmented tissue",
            "melanin pigment",
            "foamy macrophages",
            "cholesterol clefts",
        ],

        # Cellular and nuclear morphology
        "cellular_nuclear": [
            "dense nuclei",
            "sparse nuclei",
            "high nuclear density",
            "low nuclear density",
            "large atypical nuclei",
            "small uniform nuclei",
            "pleomorphic nuclei",
            "hyperchromatic nuclei",
            "vesicular nuclei",
            "prominent nucleoli",
            "mitotic figures",
            "apoptotic bodies",
            "tumor cells with enlarged nuclei",
            "spindle-shaped cells",
            "round tumor cells",
            "small blue cells",
            "inflammatory cells",
            "lymphocytes",
            "plasma cells",
            "neutrophils",
            "macrophages",
            "multinucleated giant cells",
            "cell-level morphology",
            "nuclear-level detail",
            "high magnification cellular morphology",
        ],

        # Tissue structures that are useful across many DP tasks
        "anatomical_structures": [
            "basement membrane-like structures",
            "capsule-like fibrous tissue",
            "nerve bundles",
            "perineural tissue",
            "vascular invasion-like morphology",
            "lymphovascular spaces",
            "fatty tissue invasion",
            "smooth muscle bundles",
            "skeletal muscle tissue",
            "cartilage-like tissue",
            "bone-like tissue",
            "skin adnexal structures",
            "surface epithelium",
            "ulcerated surface tissue",
            "cystic spaces",
            "microcystic spaces",
        ],

        # Scale and field-of-view descriptors
        "scale_context": [
            "low magnification tissue architecture",
            "medium magnification tissue architecture",
            "high magnification cellular morphology",
            "large field of view tissue structure",
            "small field of view cellular detail",
            "tissue-level organization",
            "cell-level morphology",
            "gland-level architecture",
            "nuclear-level detail",
            "regional tumor architecture",
            "local cellular texture",
        ],
    },

    # Negative / quality / artifact concepts.
    # These should usually be used for filtering or down-weighting, not oversampling.
    "negative": {
        "artifact_quality": [
            "empty background",
            "white background",
            "blank glass slide background",
            "out-of-focus tissue",
            "blurry histopathology image",
            "low resolution tissue image",
            "poor quality tissue",
            "tissue fold artifact",
            "torn tissue artifact",
            "crushed tissue artifact",
            "air bubble artifact",
            "scanner artifact",
            "pen mark artifact",
            "staining artifact",
            "uneven staining",
            "dark overstained tissue",
            "pale understained tissue",
            "tissue with strong color variation",
            "tissue section chatter artifact",
            "tissue section knife mark artifact",
            "dust or debris on slide",
            "blood only",
            "mostly background with little tissue",
            "necrotic debris without viable tissue",
        ],
    },

    # Optional analysis concepts.
    # These are useful for auditing tissue identity bias, but should not be the main sampling target.
    "analysis": {
        "tissue_identity": [
            "breast tissue",
            "prostate tissue",
            "colorectal tissue",
            "colon tissue",
            "rectal tissue",
            "lung tissue",
            "gastric tissue",
            "kidney tissue",
            "liver tissue",
            "pancreatic tissue",
            "ovarian tissue",
            "uterine tissue",
            "endometrial tissue",
            "brain tissue",
            "skin tissue",
            "lymph node tissue",
            "thyroid tissue",
            "bladder tissue",
            "testicular tissue",
            "adrenal tissue",
            "soft tissue",
            "bone marrow tissue",
        ],
    },
}


def normalize_concept_id(text: str) -> str:
    """Convert a concept string into a stable concept id."""
    keep = []
    for ch in text.lower().strip():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "/", "_"}:
            keep.append("_")
    concept_id = "".join(keep)
    while "__" in concept_id:
        concept_id = concept_id.replace("__", "_")
    return concept_id.strip("_")


def generate_prompt_records() -> List[Dict[str, Any]]:
    """Generate prompt records with metadata."""
    records: List[Dict[str, Any]] = []
    seen_prompts: set[tuple[str, str, str]] = set()

    for polarity, category_dict in CONCEPTS.items():
        for category, concepts in category_dict.items():
            for concept in concepts:
                concept_id = normalize_concept_id(concept)

                for template_id, template in enumerate(TEMPLATES):
                    prompt = template.format(concept=concept)
                    deduplication_key = (polarity, category, prompt)

                    if deduplication_key in seen_prompts:
                        continue
                    seen_prompts.add(deduplication_key)

                    records.append(
                        {
                            "prompt_id": f"{polarity}__{category}__{concept_id}__t{template_id}",
                            "polarity": polarity,
                            "category": category,
                            "concept": concept,
                            "concept_id": concept_id,
                            "template_id": template_id,
                            "template": template,
                            "prompt": prompt,
                        }
                    )

    return records


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "prompt_id",
        "polarity",
        "category",
        "concept",
        "concept_id",
        "template_id",
        "template",
        "prompt",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_txt(records: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(row["prompt"] + "\n")


def write_concepts_json(path: Path) -> None:
    """Save the raw concept bank for reproducibility."""
    payload = {
        "templates": TEMPLATES,
        "concepts": CONCEPTS,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./prompt_bank",
        help="Output directory.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = generate_prompt_records()

    write_jsonl(records, out_dir / "prompts.jsonl")
    write_csv(records, out_dir / "prompts.csv")
    write_txt(records, out_dir / "prompts.txt")
    write_concepts_json(out_dir / "concept_bank.json")

    n_positive = sum(1 for r in records if r["polarity"] == "positive")
    n_negative = sum(1 for r in records if r["polarity"] == "negative")
    n_analysis = sum(1 for r in records if r["polarity"] == "analysis")

    unique_concepts = sorted({r["concept_id"] for r in records})

    print(f"Saved prompt bank to: {out_dir.resolve()}")
    print(f"Total prompts: {len(records)}")
    print(f"Unique concepts: {len(unique_concepts)}")
    print(f"Positive prompts: {n_positive}")
    print(f"Negative prompts: {n_negative}")
    print(f"Analysis prompts: {n_analysis}")
    print(f"Files:")
    print(f"  - {out_dir / 'prompts.jsonl'}")
    print(f"  - {out_dir / 'prompts.csv'}")
    print(f"  - {out_dir / 'prompts.txt'}")
    print(f"  - {out_dir / 'concept_bank.json'}")


if __name__ == "__main__":
    main()
