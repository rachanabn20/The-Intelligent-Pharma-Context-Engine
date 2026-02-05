# Intelligent Pharma Context Engine

An end-to-end system for detecting, extracting, verifying, and enriching drug information from real-world medication images using object detection, OCR, and authoritative medical databases.

---

## Project Overview

Medication images are difficult to interpret reliably due to glare, curved surfaces, dense layouts, and inconsistent packaging. OCR-only approaches are unsafe for healthcare use because they often produce incorrect or hallucinated drug names.

This project addresses the problem by grounding all extracted text in trusted medical knowledge bases, prioritizing semantic correctness over raw OCR accuracy. When verification confidence is insufficient, the system explicitly returns `Unknown`.

---

## What This Repository Contains

- `the_intelligent_pharma_context_engine.ipynb`  
  A complete, runnable notebook implementing the full pipeline:
  - Medicine region detection (YOLOv8)
  - Region-based OCR with multiple preprocessing strategies
  - Text normalization and fragment generation
  - Drug verification using FDA Drug Labels and RxNorm
  - Dosage and barcode extraction
  - Confidence scoring and conservative rejection
  - Evaluation with Character Error Rate (CER) and Entity Match Rate

- `README.md`  
  Project description, datasets, and results summary.

---

## Pipeline Summary

1. Detect medicine regions using YOLOv8  
2. Apply OCR on detected regions and full image  
3. Generate robust text fragments from OCR output  
4. Match and validate drug candidates using FDA and RxNorm  
5. Extract dosage and barcode information conservatively  
6. Compute verification confidence  
7. Report results or return `Unknown` if confidence is low  

---

## Extracted Information

- Drug name  
- Manufacturer  
- Dosage (if reliably detectable)  
- Barcode or NDC (if present)  

All outputs are verified against authoritative databases before acceptance.

---

## Datasets Used

- **FDA Drug Labels (openFDA)**  
  https://open.fda.gov/apis/drug/label/

- **Pills Inside Bottles (OCR targets)**  
  https://huggingface.co/datasets/gwenxin/pills_inside_bottles

- **Medicine Bottle Object Detection Dataset**  
  https://universe.roboflow.com/project-ko6pf/medicine-bottle

- **RxNorm Database**  
  https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html

---

## Evaluation Metrics

### Character Error Rate (CER)

CER = (S + D + I) / N

Computed against the ground-truth drug name when available. When ground truth is unavailable, CER is reported for analysis but excluded from correctness scoring.

### Entity Match Rate

- Evaluated only on verifiable entities  
- `Unknown` outputs are excluded  
- Measures semantic correctness rather than OCR similarity  

---

## Results

Evaluation on 20 mixed medication and bottle images:

- Entity Match Rate (evaluated only): **0.96**
- Drug name accuracy: **1.00**
- Manufacturer accuracy: **1.00**
- Dosage accuracy: **0.00** (limited ground truth)
- Mean Character Error Rate: **0.80**
- Mean verification confidence: **0.21**
- Mean processing time per image: **3.48 seconds**
- Runtime target under 12 seconds: **Pass**

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

This project demonstrates that knowledge-grounded verification is essential for safe medication understanding. By validating OCR output against FDA and RxNorm, the system achieves high semantic accuracy while explicitly representing uncertainty.
