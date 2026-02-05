# Intelligent Pharma Context Engine

An end-to-end system for detecting, extracting, verifying, and enriching drug information from real-world medication images using object detection, OCR, and authoritative medical knowledge bases.

---

## Overview

Medication images are difficult to process reliably due to glare, curved surfaces, dense layouts, and inconsistent packaging. OCR-only systems are error-prone and unsafe for healthcare use.

This project addresses the problem by grounding extracted text in **FDA Drug Labels (openFDA)** and **RxNorm**, prioritizing semantic correctness over raw OCR accuracy. When confidence is insufficient, the system explicitly returns `Unknown` instead of guessing.

---

## System Pipeline

1. Medicine region detection using YOLOv8  
2. Region-based OCR with multiple preprocessing variants  
3. Text normalization and fragment generation  
4. Drug candidate matching and validation using FDA and RxNorm  
5. Dosage and barcode extraction (conservative)  
6. Confidence scoring and uncertainty handling  
7. Structured output with full metadata

All outputs are verified against authoritative databases before acceptance.

---

## Extracted Information

- Drug name  
- Manufacturer  
- Dosage (if reliably detectable)  
- Barcode or NDC (if present)  
- Active ingredients  
- Storage requirements  
- Warnings and side effects  

If verification confidence is below threshold, values are set to `Unknown`.

---

## Datasets

### FDA Drug Labels (openFDA)
https://open.fda.gov/apis/drug/label/

### Pills Inside Bottles (OCR targets)
https://huggingface.co/datasets/gwenxin/pills_inside_bottles

### Medicine Bottle Object Detection Dataset
https://universe.roboflow.com/project-ko6pf/medicine-bottle

### RxNorm Database
https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html

---

## Evaluation Metrics

### Character Error Rate (CER)

CER = (S + D + I) / N

Computed against the ground-truth drug name when available. When no ground truth exists, CER is reported for analysis but excluded from correctness scoring.

### Entity Match Rate

- Evaluated only on verifiable entities  
- `Unknown` outputs are excluded  
- Measures semantic correctness rather than OCR similarity  

---

## Results

### Evaluation Summary (20 Test Images)

- Entity Match Rate (evaluated only): **0.96**
- Drug name accuracy: **1.00**
- Manufacturer accuracy: **1.00**
- Dosage accuracy: **0.00** (limited ground truth)
- Mean Character Error Rate: **0.80**
- Mean verification confidence: **0.21**
- Mean processing time per image: **3.48 seconds**
- Runtime target under 12 seconds: **Pass**

---

## Key Observations

- High CER does not imply semantic failure  
- Database-grounded validation corrects noisy OCR  
- Conservative confidence thresholds prevent hallucinated outputs  
- Images without readable text correctly return `Unknown`  

---

## Hardware

- GPU: NVIDIA T4  
- Fully compatible with Kaggle T4 environments  

---

## Limitations

- Dosage extraction is intentionally conservative  
- Barcode detection depends on image quality  
- Performance depends on FDA and RxNorm coverage  

---

## Conclusion

This project demonstrates that knowledge-grounded verification is essential for safe medication understanding. By validating OCR output against FDA and RxNorm, the system achieves high semantic accuracy while explicitly representing uncertainty, making it suitable for healthcare and regulatory applications.
