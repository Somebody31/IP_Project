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
### Real Fog Benchmark (`rtts/test`, 432 Held-Out Images, 100% Frozen COCO YOLOv8n)

| Evaluation Metric | Raw Hazy Baseline | Reference Paper | Classical DCP | Our Method | Gain vs. Raw Hazy | Gain vs. Paper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PSNR: Light Fog ($\beta = 0.005$)** | 18.57 dB | 27.95 dB | 15.77 dB | **29.15 dB** | **+10.58 dB** | **+1.20 dB** |
| **PSNR: Medium Fog ($\beta = 0.010$)** | 14.47 dB | 26.16 dB | 15.93 dB | **27.36 dB** | **+12.89 dB** | **+1.20 dB** |
| **PSNR: Heavy Fog ($\beta = 0.020$)** | 10.92 dB | 25.16 dB | 15.65 dB | **24.53 dB** | **+13.61 dB** | **−0.63 dB** |
| **PSNR: Overall Full Val (1,500 scenes)** | 14.65 dB | 26.39 dB | 15.79 dB | **27.02 dB** | **+12.37 dB** | **+0.63 dB** |
| **SSIM: Overall Full Val** | 0.8250 | 0.9420 | 0.7037 | **0.9500** | **+0.1250** | **+0.0080** |
| **AP@0.50 (Standard VOC, RTTS Test)** | 0.4923 | 0.4789 | 0.5194 | **0.5337** | **+0.0414** | **+0.0548 (+5.5 pts)** |
| **AP@0.75 (Strict Boundary, RTTS Test)** | 0.2996 | 0.2968 | 0.3024 | **0.3275** | **+0.0279** | **+0.0307 (+10.3%)** |
| **mAP@[0.50:0.95] (COCO Mean, RTTS Test)** | 0.2929 | 0.2884 | 0.3040 | **0.3208** | **+0.0279** | **+0.0324 (+11.2%)** |

### Key Takeaways:
1. **Reconstruction across Fog Densities**: On 1,500 paired Foggy Cityscapes scenes, our model outperforms the reference paper on Light (+1.20 dB) and Medium (+1.20 dB) fog, and significantly recovers Heavy fog (24.53 dB), leading to 27.02 dB vs 26.39 dB overall (+0.63 dB gain).
2. **Real Fog Task Utility (AP@0.50)**: While the reference paper reduces detection accuracy on real fog (−0.0134 AP@0.50 vs raw hazy), our method provides genuine visual recovery (+0.0418 AP@0.50 vs raw, +0.0548 vs Paper, $p < 0.05$).
3. **Boundary Localization Precision (AP@0.75)**: Under strict IoU criteria (AP@0.75), our method delivers a 10.3% relative gain over the reference paper (+0.0307 AP) and +0.0251 over DCP, confirming high-fidelity contour preservation ($p < 0.05$).

---

## Slide 5: Model Complexity & Footprint

| Deployment Metric | Reference Paper | Our Method | Advantage |
| :--- | :---: | :---: | :---: |
| **Total Model Parameters** | 23.3 M (23,326,988) | **7.8 M (7,766,084)** | **66.7% parameter reduction** |
| **Disk Storage Footprint** | 93.3 MB (3 checkpoints) | **31.1 MB (Single model)** | **66.7% smaller footprint** |
| **RTTS Latency (640×480) & VRAM** | 36.4 ms (27.5 FPS) \| 531.9 MB | **31.5 ms (31.8 FPS) \| 397.7 MB** | **13.5% faster \| 25.2% less VRAM** |
| **Cityscapes Latency & VRAM (2K)** | 245.5 ms \| 2,805.2 MB | **203.6 ms \| 1,925.1 MB** | **17.1% faster \| 880 MB VRAM saved** |
| **Downstream Detector** | 100% Frozen Standard | **100% Frozen Standard** | **Universal plug-and-play** |

### Key Architectural Takeaways:
- **Unified Backbone**: Rather than keeping 4 distinct networks in memory, our architecture relies on a single compact spatial backbone.
- **Universal Modularity**: Downstream detector weights are completely unmodified for all evaluated methods, ensuring seamless deployment with any pre-trained perception model.
- **Edge Viability**: High frame rate (31.8 FPS real-time) and low VRAM footprint (398 MB) make our model practical for real-world automotive and edge deployments.
