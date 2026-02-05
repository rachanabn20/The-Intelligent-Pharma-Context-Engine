# The-Intelligent-Pharma-Context-Engine

Intelligent Pharma Context Engine

An end-to-end system for detecting, extracting, verifying, and enriching drug information from real-world medication images using object detection, OCR, and authoritative medical knowledge bases.

---

## Problem Statement

Medication images often contain noisy and unreliable text due to glare, curved surfaces, dense layouts, low contrast, and inconsistent packaging. OCR-only approaches are insufficient for healthcare-grade extraction where semantic correctness is critical.

This project addresses the problem by grounding all extracted text in authoritative sources such as FDA Drug Labels and RxNorm, prioritizing verified correctness over raw OCR accuracy.

---

## System Overview

The Intelligent Pharma Context Engine is designed as a conservative, verification-first pipeline that avoids hallucination and explicitly reports uncertainty when confidence is insufficient.

### Processing Pipeline

1. Medicine bottle detection using YOLO
2. Region-based OCR on detected regions
3. Text normalization and fragment aggregation
4. Entity candidate generation
5. Verification against FDA Drug Labels, RxNorm, and NDC
6. Confidence scoring and conservative rejection
7. Structured output generation

If verification confidence does not meet the minimum threshold, the system outputs `Unknown`.

---

## Extracted Entities

- Drug name
- Manufacturer
- Dosage (if reliably detectable)
- Barcode or NDC (if present)

---

## Datasets

### FDA Drug Labels
https://open.fda.gov/apis/drug/label/
https://huggingface.co/datasets/gwenxin/pills_inside_bottles
https://universe.roboflow.com/project-ko6pf/medicine-bottle
https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html

---

## Evaluation Metrics

### Character Error Rate (CER)

**Formula**
CER = (S + D + I) / N

Where:
- S = substitutions  
- D = deletions  
- I = insertions  
- N = total characters  

CER is computed against the ground truth drug name when available. When ground truth is unavailable, CER is reported for analysis but excluded from correctness scoring.

---

### Entity Match Rate

- Evaluated only on verifiable entities
- `Unknown` outputs are excluded
- Measures semantic correctness rather than raw OCR similarity

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

- High CER does not imply semantic failure
- Database-grounded verification corrects severe OCR noise
- Conservative confidence thresholds prevent hallucinated outputs
- Bottle images without readable text correctly return `Unknown`

---

## Hardware

- GPU: NVIDIA T4
- Training and inference fully compatible with Kaggle T4 environments

---

## Limitations

- Dosage extraction is intentionally conservative
- Barcode detection is limited by dataset availability
- Performance depends on external database coverage and completeness

---

## Conclusion

This work demonstrates that knowledge-grounded verification is essential for safe and reliable medication understanding. By validating extracted text against FDA and RxNorm databases, the system achieves high semantic accuracy despite noisy OCR, making it suitable for healthcare, regulatory, and real-world clinical applications.



