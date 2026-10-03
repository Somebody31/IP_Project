# Presentation: Real-World Image Dehazing for Object Detection
### Plain & Minimalist Comparative Analysis

---

## Slide 1: Title
### Real-World Image Dehazing for Object Detection
**Comparative Analysis: Performance, Visual Distinctions, and Model Complexity**

- **Methods Evaluated**:
  1. Raw Hazy Baseline (Untreated)
  2. Reference Paper (Density-Routed CORUN)
  3. Classical DCP (He et al., 2009)
  4. Our Method (Unified Spatial Network)
- **Target Task**: Downstream Object Detection (Off-the-shelf Frozen COCO YOLOv8n)
- **Evaluation Split**: Held-out `rtts/valid` (865 real fog scenes, 8,659 ground truth objects)

---

## Slide 2: Method Overview

| Method | Architecture / Pipeline | Core Mechanism | Primary Limitation |
| :--- | :--- | :--- | :--- |
| **1. Raw Hazy Baseline** | No Preprocessing | Camera sensor image fed directly to the detector | Severe loss of contrast; distant objects and fine boundaries veiled by atmospheric scattering. |
| **2. Reference Paper** | 4 Separate Networks (1 Classifier + 3 U-Nets) | Classifier assigns image to Light, Medium, or Heavy fog branch | Assumes uniform fog across the image; partitions dataset into 3 smaller subsets; over-darkens foregrounds. |
| **3. Classical DCP** | Zero-Shot Physical Prior (He et al., 2009) | Dark Channel Prior + Guided Filter estimates transmission & airlight | Heuristic breaks down in sky, high-reflectance regions, and headlights, producing severe halos and color casts. |
| **4. Our Method** | Single Unified Network (7.8 M parameters) | Continuous per-pixel spatial gating that adapts across scene depth | None of the above: trained with realistic airlight randomization ($A \in [0.4, 0.95]$) and frozen perception guidance. |

---

## Slide 3: Visual Comparison on Real Fog
### Distinguishable Visual Characteristics

![Direct Visual Dehazing Quality](file:///home/sathwik/Dev/Projects/IP_Project/docs/figures/presentation/method_comparison_visual.png)

![Downstream YOLOv8n Object Detection](file:///home/sathwik/Dev/Projects/IP_Project/docs/figures/presentation/method_comparison_detection.png)

### Key Distinguishing Traits:
- **(a) Raw Hazy**: Pale, low-contrast haze veil across the scene; distant structures and vehicles are obscured (5 detections).
- **(b) Reference Paper**: Severe sky over-darkening and blotchy dark halos around buildings and lamp posts; contrast distortion reduces detection accuracy (4 detections).
- **(c) Classical DCP**: Unnatural yellow/greenish sky discoloration and blocking artifacts near high-radiance regions (4 detections).
- **(d) Our Method**: Natural sky tone, balanced illumination, clean structural edges, and the highest downstream detection yield (7 detections).

---

## Slide 4: Quantitative Comparison (Method Values)
### Held-Out Real Fog Benchmark (`rtts/valid`, 865 Images, Frozen YOLOv8n)

| Evaluation Metric | Raw Hazy Baseline | Reference Paper | Classical DCP | Our Method | Gain vs. Raw Hazy | Gain vs. Paper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AP@0.50 (Standard VOC)** | 0.4527 | 0.4415 | 0.4706 | **0.4891** | **+0.0364** | **+0.0476 (+4.8 pts)** |
| **AP@0.75 (Strict Boundary)** | 0.2593 | 0.2602 | 0.2825 | **0.2903** | **+0.0310** | **+0.0301 (+11.6%)** |
| **AP@0.95 (Ultra-Strict)** | 0.0163 | 0.0212 | 0.0209 | **0.0215** | **+0.0052** | **+0.0003 (+1.4%)** |
| **mAP@[0.50:0.95] (COCO Mean)** | 0.2657 | 0.2625 | 0.2811 | **0.2924** | **+0.0267** | **+0.0299 (+11.4%)** |
| **Total Detections** | 83,105 | 100,073 | 99,945 | **98,097** | **+14,992** | **−1,976 (Cleaner)** |

### Key Takeaways:
1. **Positive Task Transfer**: The reference paper degrades detection accuracy below untreated images (−0.0112 AP@0.50), whereas our method consistently improves performance (+0.0364 AP@0.50).
2. **Beating Classical DCP**: For the first time, our neural model systematically outperforms classical DCP across every IoU threshold (+0.0185 AP@0.50, +0.0113 mAP).
3. **Contour Precision**: At strict IoU = 0.75, our method achieves an **11.6% relative gain over the paper** while reducing false-positive clutter.

---

## Slide 5: Model Complexity & Footprint

| Deployment Metric | Reference Paper | Our Method | Advantage |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 26.4 M (26,378,804) | **7.8 M (7,766,084)** | **70.6% parameter reduction** |
| **Active Models Hosted** | 4 (1 Classifier + 3 Dehazers) | **1 (Single Unified Model)** | **3.4× lower memory footprint** |
| **Inference Latency** | 240 ms / image | **230 ms / image** | Faster runtime |
| **Downstream Detector** | Fine-tuned / Coupled | **100% Frozen COCO Standard** | **Universal plug-and-play** |

### Key Architectural Takeaways:
- **Unified Backbone**: Rather than keeping 4 distinct networks in memory, our architecture relies on a single compact spatial backbone.
- **Universal Modularity**: Downstream detector weights are completely unmodified, ensuring seamless deployment with any pre-trained perception model.
- **Edge Viability**: Lower parameters and faster execution make our model practical for real-world automotive and surveillance applications.
