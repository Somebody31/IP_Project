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
### Distinguishable Visual Characteristics (Illustrative Example)

![Direct Visual Dehazing Quality](file:///home/sathwik/Dev/Projects/IP_Project/docs/figures/presentation/method_comparison_visual.png)

![Downstream YOLOv8n Object Detection](file:///home/sathwik/Dev/Projects/IP_Project/docs/figures/presentation/method_comparison_detection.png)

### Distinguishing Characteristics:
- **(a) Raw Hazy**: Pale, low-contrast haze veil across the scene; distant structures and vehicles are obscured (5 detections).
- **(b) Reference Paper**: Severe sky over-darkening and blotchy dark halos around buildings and lamp posts; contrast distortion reduces detection accuracy (4 detections).
- **(c) Classical DCP**: Unnatural yellow/greenish sky discoloration and blocking artifacts near high-radiance regions (4 detections).
- **(d) Our Method**: Natural sky tone, balanced illumination, clean structural edges, and the highest downstream detection yield (7 detections).

---

## Slide 3B: Qualitative Verification: Multi-Scene Random Sample Grid
### Systematic Non-Selective Evaluation across Diverse Fog Densities

![Multi-Scene Random Evaluation Grid](file:///home/sathwik/Dev/Projects/IP_Project/docs/figures/presentation/random_samples_grid.png)

- **Unbiased Selection**: 4 test scenes randomly drawn from `rtts/valid` using fixed seed 42 across all 4 methods.
- **Consistent Visual Behavior**: Confirms across varying scenes that Classical DCP consistently suffers from sky discoloration, Reference Paper creates patchy dark regions under non-uniform fog, and Our Method maintains stable color temperature and natural contrast.

---

## Slide 4: Quantitative Comparison (Method Values)
### Held-Out Real Fog Benchmark (`rtts/valid`, 865 Images, Frozen YOLOv8n)

| Evaluation Metric | Raw Hazy Baseline | Reference Paper | Classical DCP | Our Method | Gain vs. Raw Hazy | Gain vs. Paper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR: Light Fog ($\beta = 0.005$)** | 18.57 dB | 27.95 dB | 15.77 dB | **28.97 dB** | **+10.40 dB** | **+1.02 dB** |
| **PSNR: Medium Fog ($\beta = 0.010$)** | 14.47 dB | 26.16 dB | 15.93 dB | **27.04 dB** | **+12.57 dB** | **+0.88 dB** |
| **PSNR: Heavy Fog ($\beta = 0.020$)** | 10.92 dB | 25.16 dB | 15.65 dB | **24.11 dB** | **+13.19 dB** | **−1.05 dB** |
| **PSNR: Overall Full Val (1,500 scenes)** | 14.65 dB | 26.42 dB | 15.79 dB | **26.71 dB** | **+12.06 dB** | **+0.29 dB** |
| **SSIM: Overall Full Val** | 0.8250 | 0.9482 | 0.7037 | **0.9485** | **+0.1235** | **+0.0003** |
| **AP@0.50 (Standard VOC, RTTS)** | 0.4527 | 0.4415 | 0.4706 | **0.4891** | **+0.0364** | **+0.0476 (+4.8 pts)** |
| **AP@0.75 (Strict Boundary, RTTS)** | 0.2593 | 0.2602 | 0.2825 | **0.2903** | **+0.0310** | **+0.0301 (+11.6%)** |
| **mAP@[0.50:0.95] (COCO Mean, RTTS)** | 0.2657 | 0.2625 | 0.2811 | **0.2924** | **+0.0267** | **+0.0299 (+11.4%)** |

### Key Takeaways:
1. **Reconstruction across Fog Densities**: On 1,500 paired Foggy Cityscapes scenes, our model outperforms the reference paper on Light (+1.02 dB) and Medium (+0.88 dB) fog, but trails on Heavy fog (−1.05 dB), leading to 26.71 dB vs 26.42 dB overall.
2. **Real Fog Task Utility (AP@0.50)**: While the reference paper reduces detection accuracy on real fog (−0.0112 AP@0.50 vs raw hazy), our method provides genuine visual recovery (+0.0364 AP@0.50 vs raw, +0.0185 vs DCP).
3. **Boundary Localization Precision (AP@0.75)**: Under strict IoU criteria (AP@0.75), our method delivers an 11.6% relative gain over the reference paper (+0.0301 AP) and +0.0078 over DCP, confirming high-fidelity contour preservation.

---

## Slide 5: Model Complexity & Footprint

| Deployment Metric | Reference Paper | Our Method | Advantage |
| :--- | :---: | :---: | :---: |
| **Total Parameters** | 26.4 M (26,378,804) | **7.8 M (7,766,084)** | **70.6% parameter reduction** |
| **Active Models Hosted** | 4 (1 Classifier + 3 Dehazers) | **1 (Single Unified Model)** | **3.4× lower memory footprint** |
| **Inference Latency** | Benchmark in Progress (Phase 1) | Benchmark in Progress (Phase 1) | Unified single-pass execution |
| **Downstream Detector** | 100% Frozen Standard | **100% Frozen Standard** | **Universal plug-and-play** |

### Key Architectural Takeaways:
- **Unified Backbone**: Rather than keeping 4 distinct networks in memory, our architecture relies on a single compact spatial backbone.
- **Universal Modularity**: Downstream detector weights are completely unmodified for all evaluated methods, ensuring seamless deployment with any pre-trained perception model.
- **Edge Viability**: Lower parameters and single-model design make our model practical for real-world automotive and edge deployments.
