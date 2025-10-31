---
title: NPS SRO ML Project
titleTemplate: GNN for Physics Analysis
---

## Welcome to the NPS SRO Machine Learning Project

This project applies Graph Neural Network (GNN) methods to streaming-like data for physics analysis, focusing on clustering and classification of signal over background in the NPS (Neutral Particle Spectrometer) detector.

### Project Overview

The main objective is to leverage GNN architectures to analyze calorimeter data from the NPS detector, using the VTP (VETROC Trigger Processor) trigger information as ground truth for training.

**Key Components:**
- C++ code for converting ROOT events into graph data (*.pt format)
- PyTorch-based machine learning pipeline
- Edge classification using waveform data
- GravNet and transformer-based ConvNet architectures

### Data Information

**Data Location:** `/cache/hallc/c-nps/analysis/pass2/replays/updated`

**Key Branches:**
- `NPS.cal.fly.adcSampWaveform` - Waveforms from fADC250 in each of the 1080 blocks
- `NPS.cal.vtpClusX, Y, Time` - Position and time of the center position of VTP cluster
- `NPS.cal.fly.block_clusterID` - Clusters found by hcana

### Current Progress

The first objective focuses on **edge classification**: constructing input graphs from waveforms where positions are known, then using GravNet or transformer-based ConvNet to learn edge connectivity and compare with VTP edges.

### Resources

- VTP Trigger Manual (see documentation)
- Hall C NPS Analysis Data 
