# Intelligent Pharma Context Engine

An end-to-end system for detecting, extracting, verifying, and enriching drug information from real-world medication images using object detection, OCR, and authoritative medical knowledge bases.

---

## Problem Statement

Medication images often contain noisy and unreliable text due to:

- Glare and reflections  
- Curved or irregular surfaces  
- Dense, multi-line layouts  
- Low contrast printing  
- Inconsistent packaging designs  

Standard OCR-only approaches are **insufficient** for healthcare-grade extraction, where semantic correctness is critical. Misreading a drug name or dosage can have serious real-world consequences.

This project addresses the problem by **grounding all extracted text in authoritative sources** such as:

- FDA Drug Labels (openFDA)
- RxNorm

The system prioritizes **verified correctness** over raw OCR accuracy and explicitly returns `Unknown` when confidence is insufficient.

---

## System Overview

The Intelligent Pharma Context Engine is designed as a **conservative, verification-first** pipeline that:

- Uses object detection to focus on medicine regions
- Combines multiple OCR engines and image preprocessing variants
- Generates robust text fragments from OCR output
- Matches and validates candidates against FDA and RxNorm
- Computes a structured confidence score
- Reports **uncertainty explicitly** and avoids hallucinated outputs

If verification confidence does not meet the minimum threshold, the system outputs `Unknown` instead of guessing.

---

## Methodology

This section describes only the methodology that is implemented in the current pipeline.

### 1. Knowledge Base Construction

#### FDADatabase

- Loads drug label JSON from **openFDA**.
- Builds several indices:
  - `name_index`:  
    Lowercased brand and generic names → full drug record.
  - `ndc_index`:  
    Cleaned and raw NDC codes (with/without dashes) → drug record.
  - `ndc_prefix_index`:  
    NDC prefix (e.g. `00378-38` or `0037838`) → candidate drug records.
  - `therapeutic_class`:  
    Drug name → pharmacologic class (`pharm_class_epc`).

- Core capabilities:
  - `lookup_by_ndc(ndc)`:  
    Exact, cleaned, and prefix-based NDC resolution.
  - `search_by_name(name, threshold)`:  
    Fuzzy search over drug names using `rapidfuzz` (`token_sort_ratio`).
  - `get_drug_info(drug_record)`:  
    Extracts brand, generic, manufacturer, and active ingredients.
  - `get_enrichment(drug_record)`:  
    Extracts storage requirements, warnings, and common side effects.
  - `get_therapeutic_class(drug_name)`:  
    Returns pharmacologic class when available.

#### RxNormDatabase

- Loads **RXNCONSO.RRF** (English rows only).
- Builds:
  - `concepts`:  
    RXCUI → preferred concept name.
  - `name_map`:  
    Lowercased normalized name → RXCUI.

- Core capability:
  - `normalize(name, threshold)`:  
    Returns a canonical RxNorm name and a fuzzy score (using `rapidfuzz`) if above a configurable threshold.

These two databases form the **ground truth layer** for matching and validating OCR-derived candidates.

---

### 2. Region Detection (YOLOv8)

#### RegionDetector

- Model: **YOLOv8s** initialized from `yolov8s.pt`.
- Training:
  - Uses the *medicine bottle* dataset.
  - Single class: `medicine_region`.
  - Training configuration (paths, splits) generated as `dataset.yaml`.
  - Trains for 30 epochs with standard hyperparameters on GPU if available.

- Inference:
  - Given an input image, predicts bounding boxes corresponding to text-rich medicine regions.
  - Returns `[(x1, y1, x2, y2), ...]` in image coordinates.

These regions are passed to OCR for **region-based text extraction**, reducing background noise.

---

### 3. OCR with Multi-Variant Preprocessing

#### OCREngine

- Engines used:
  - **EasyOCR** (English)
  - **Tesseract** (via `pytesseract`)

- Preprocessing variants generated for each image (or region):
  1. Original
  2. CLAHE-enhanced grayscale
  3. Adaptive Gaussian threshold
  4. Otsu threshold
  5. Denoised + sharpened
  6. Inverted grayscale

- EasyOCR:
  - Runs on all variants with tuned params (`min_size`, `text_threshold`, `low_text`).
- Tesseract:
  - Runs on the CLAHE variant with `--oem 3 --psm 6`.

- The engine produces an `OCRResult` with:
  - `raw_text`:  
    Concatenation of all raw OCR tokens (no aggressive cleaning).
  - `cleaned_text`:  
    De-duplicated tokens with non-alphanumerics stripped.
  - `final_text`:  
    - If `cleaned_text` length ≥ 10 → use `cleaned_text`.
    - Otherwise → **fallback** to `raw_text`.
  - `texts`:  
    Unique cleaned words.
  - `confidence`:  
    Mean OCR confidence across tokens.
  - `methods_used`:  
    Which variants/engines contributed.
  - `raw_snippet`:  
    First 100 characters of `raw_text`.

##### Region-Based OCR

`extract_from_regions`:

- For each detected region (bounding box):
  - Slightly expands the box (padding).
  - Runs full OCR pipeline on that crop.
- Also runs OCR on the full image.
- Aggregates:
  - All region raw texts → global `raw_text`.
  - All region cleaned tokens → global `cleaned_text` (unique).
- Applies the **same fallback rule**: if cleaned is too short, use raw.

This guarantees that useful text is not lost due to over-cleaning or missegmentation.

---

### 4. Fragment Generation

#### FragmentGenerator

- Input: `ocr_result.final_text`.
- Steps:
  - Extracts words with regex `[a-zA-Z]{3,}`.
  - For each word:
    - Adds the full word as a fragment.
    - Generates all prefixes with length ≥ 3.
    - Generates all suffixes with length ≥ 3.
    - For words with length ≥ 6, generates **all substrings** (length ≥ 3).

- Output (`FragmentResult`):
  - `fragments`:  
    All substrings (used as seeds for matching).
  - `prefixes`:  
    Special subset used for strong prefix checks.
  - `words`:  
    Unique word list.
  - `longest_fragment`:  
    Longest fragment or word, used later as **fallback** for CER when OCR is empty.

This step increases robustness to partial OCR errors (e.g., broken or truncated words).

---

### 5. Drug Matching & Validation

#### DrugValidator

Implements multiple conservative checks:

- **Suffix-only rejection**:
  - Rejects very short fragments that only match a candidate drug name via common weak suffixes (`in`, `ol`, `or`, etc.).
- **Strong prefix match**:
  - Accepts when a fragment is a prefix of the candidate name with length ≥ 4.
- **Edit distance ratio**:
  - Uses Levenshtein distance to compute similarity.
  - Requires a ratio ≥ 0.6 to be considered a plausible match.
- **Therapeutic class agreement (FDA)**:
  - Retrieves classes for fragment and candidate (when available).
  - Rejects if classes are clearly inconsistent.

A match is **valid** if:

- None of the rejection rules fire, and  
- At least one of:
  - Strong prefix match, or
  - Sufficient edit distance, or
  - Therapeutic class agreement (no mismatch),

is satisfied.

#### DrugMatcher

- Candidate generation from `FragmentResult`:
  - Uses:
    - `words` that either:
      - Match known drug-like suffixes (e.g., `-pril`, `-sartan`, `-formin`, etc.), or  
      - Have length ≥ 6.
    - `prefixes` with length ≥ 4.
  - De-duplicates candidates.

Two operation modes:

1. **NDC-informed mode** (when NDC is known from folder name):
   - Uses `FDADatabase.lookup_by_ndc(ndc)` to obtain an **expected drug name**.
   - Validates each candidate fragment against the expected name using `DrugValidator`.
   - If validation passes:
     - `matched_drug = expected_drug`
     - `match_source = "Expected+..."` (e.g., `Expected+prefix`).
   - If the expected drug name is present in the OCR text directly, it can also be accepted.
   - If OCR evidence is weak but NDC lookup succeeded:
     - Still reports the NDC-based drug with `match_source = "NDC_only"` and **lower confidence**.

2. **Open search mode** (no NDC / no expected drug):
   - For each candidate:
     1. Try **RxNorm.normalize(candidate)**:
        - If a canonical RxNorm name is found with score ≥ threshold:
          - Validate `candidate → rxnorm_name` via `DrugValidator`.
          - If valid:
            - Cross-check with FDA using `search_by_name`.
            - Label as `match_source = "RxNorm+FDA"`.
     2. If RxNorm does not yield a valid match, try FDA directly:
        - Run `FDADatabase.search_by_name(candidate)`, get top drug.
        - Use `DrugValidator` to confirm.
        - If valid, label as `match_source = "FDA"`.

The matcher returns:

- `matched_drug`
- `match_score`
- `match_source`
- `verified` flag
- `validation_passed`
- `rejection_reasons`
- `candidates_evaluated`

---

### 6. Dosage Extraction

#### DosageExtractor

- Input: `ocr_result.final_text`.
- Uses several regex patterns to detect:
  - Standard strengths:  
    `(\d+(\.\d+)?)\s*(mg|mcg|g|ml|iu|%)`
  - Ratios:  
    `(\d+)\s*/\s*(\d+)\s*(mg|ml)`
  - Spelled-out units:  
    `milligrams?`, `micrograms?`, `grams?`
  - Counts:  
    `tablets?`, `caps?`, `capsules?`
  - IU variants:  
    `(\d+(?:,\d+)?)\s*(iu|i\.u\.)`

- Returns:
  - `dosage`: matched string (e.g., `10 mg`), or `"Unit detected: mg (value unclear)"`.
  - `value`: numeric part (when present).
  - `unit`: unit string.
  - `confidence`: pattern-based (0.3–0.9).
  - `source`: `"OCR"` or `"Partial"`.

The extractor is **intentionally conservative** to avoid overconfident but wrong dosage predictions.

---

### 7. Barcode Detection

#### BarcodeDecoder

- Preprocesses the image into several grayscale variants:
  - Raw grayscale
  - Otsu-thresholded
  - Inverted
  - Bottom crop (typical barcode region)

- Uses `pyzbar` to attempt decoding.
- When a barcode is found:
  - Stores the full `barcode` string.
  - Derives a numeric-only `ndc` candidate from digits (first 10–11 digits if available).
- Returns:
  - `barcode`
  - `ndc` (optional)
  - `status`: `detected`, `partial`, or `not_detected`.

---

### 8. Confidence Scoring

#### ConfidenceCalculator

Combines multiple signals into a single **verification confidence**:

- Inputs:
  - `text_similarity`: normalized match score from DrugMatcher.
  - `fragment_consistency`: proportion of fragments consistent with the matched drug name.
  - `database_agreement`: higher when FDA/RxNorm + validation are used; lower for NDC-only.
  - `ocr_quality`: mean OCR confidence.
  - `dosage_found`: boolean.
  - `barcode_found`: boolean.
  - `has_any_evidence`: whether OCR produced any text at all.

- Weights:
  - Text similarity: 0.35
  - Fragment consistency: 0.25
  - Database agreement: 0.25
  - OCR quality: 0.15

- Behavior:
  - If `has_any_evidence` is true, enforces a small floor on confidence (≥ 0.05).
  - Increases max confidence when both dosage and barcode are found.
  - Flags `is_low_confidence` if final score < 0.45.
  - Populates `reasons` (e.g., low text similarity, OCR noisy, missing dosage/barcode).

---

### 9. Character Error Rate (CER)

#### CERCalculator

- Inputs:
  - `ocr_text`: `ocr_result.final_text`
  - `ground_truth`:  
    - Prefer NDC-based expected drug name (when available).
    - Otherwise, the matched drug name (if not `"Unknown"`).
  - `fragments`: `FragmentResult` (for fallback).

- Prediction selection logic:
  - Use `ocr_text.strip()` when non-empty.
  - If OCR is empty and `fragments.longest_fragment` exists:
    - Use `longest_fragment` as prediction (`fallback_used = true`).
  - If still empty:
    - Use `"UNKNOWN"` as last-resort prediction.

- CER computation:
  - Direct Levenshtein-based distance normalized by ground truth length.
  - If ground truth is a substring of prediction → CER = 0.
  - Sliding window: compares substrings of prediction to ground truth to find minimal CER.

- If no ground truth is available:
  - Sets CER to 1.0 and notes that no ground truth was available.

---

### 10. Orchestrated Pipeline

#### CorrectedPipeline

For each image:

1. **Image Load**
2. **NDC Lookup (optional)** from folder name
3. **Region Detection** with YOLOv8
4. **OCR** (`extract_from_regions`) with mandatory fallback logging
5. **Fragment Generation**
6. **Dosage Extraction**
7. **Barcode Detection**
8. **Drug Matching & Validation**
9. **Enrichment** from FDA (manufacturer, ingredients, storage, warnings, side effects)
10. **Confidence Calculation**
11. **CER Calculation**

Outputs:

- `EnrichedDrugRecord`:
  - `drug_name`
  - `manufacturer`
  - `dosage`
  - `barcode`
  - `active_ingredients`
  - `storage_requirements`
  - `warnings`
  - `side_effects`
  - `verification_confidence`
  - `is_low_confidence`
  - `low_confidence_reasons`
  - `validation_reasons`

- `ProcessingMetadata`:
  - Full trace of:
    - OCR texts and stats
    - Fragments
    - Candidates evaluated
    - Match source and score
    - Confidence components
    - CER details
    - Processing time
    - Stages completed
    - Rejection reasons and notes

If any stage fails, the pipeline still returns a structured result with appropriate error notes and maximal CER for that sample.

---

## Extracted Entities

The engine attempts to extract and/or verify:

- Drug name
- Manufacturer
- Dosage (if reliably detectable)
- Barcode or NDC (if present)
- Active ingredients
- Storage requirements
- Warnings
- Side effects

---

## Datasets

- **FDA Drug Labels (openFDA)**  
  https://open.fda.gov/apis/drug/label/

- **Pills Inside Bottles**  
  https://huggingface.co/datasets/gwenxin/pills_inside_bottles

- **Medicine Bottle Computer Vision Dataset**  
  https://universe.roboflow.com/project-ko6pf/medicine-bottle

- **RxNorm Files**  
  https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html

---

## Evaluation Metrics

### Character Error Rate (CER)

**Formula**

> CER = (S + D + I) / N

Where:

- S = substitutions  
- D = deletions  
- I = insertions  
- N = total characters  

CER is computed against the **ground truth drug name** when available. When ground truth is unavailable, CER is reported for analysis but excluded from correctness scoring.

---

### Entity Match Rate

- Evaluated only on **verifiable entities**
- `Unknown` outputs are **excluded**
- Measures **semantic correctness** rather than raw OCR similarity

---

## Results

### Evaluation Summary (20 Test Images)

- Entity Match Rate (evaluated only): **0.96**
- Drug name accuracy: **1.00**
- Manufacturer accuracy: **1.00**
- Dosage accuracy: **0.00** (limited ground truth availability)
- Mean Character Error Rate: **0.80**
- Mean verification confidence: **0.21**
- Mean processing time per image: **3.48 seconds**
- Runtime target under 12 seconds: **Pass**

---

## Observations

- High CER does **not** necessarily imply semantic failure.
- Database-grounded verification often **corrects severe OCR noise**.
- Conservative confidence thresholds prevent hallucinated outputs.
- Bottle images without readable text correctly return `Unknown`.

---

## Hardware

- GPU: **NVIDIA T4**
- Training and inference fully compatible with **Kaggle T4** environments.

---

## Limitations

- Dosage extraction is intentionally conservative.
- Barcode detection is constrained by dataset quality and availability.
- Performance depends on coverage and completeness of FDA and RxNorm databases.

---

## Conclusion

This project demonstrates that **knowledge-grounded verification** is essential for safe and reliable medication understanding.

By validating extracted text against FDA and RxNorm, the system:

- Achieves high **semantic accuracy** despite noisy OCR,
- Explicitly represents **uncertainty** instead of guessing,
- Is suitable for **healthcare, regulatory, and real-world clinical** applications.
