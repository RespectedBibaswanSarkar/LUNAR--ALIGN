# Legacy Code

This folder contains the original super-resolution and demonstration components of the Chandrayaan-2 prototype.

## Original purpose
The legacy repository was designed to train an ESRGAN-based super-resolution model that converts low-resolution TMC-2 lunar imagery into an OHRC-like high-resolution output. It also included a demonstration notebook and map-generation experiments for a global lunar atlas.

## Why it is no longer in the main pipeline
The new LUNAR-ALIGN architecture is focused on robust registration of lunar imagery rather than image enhancement. In the new system, the primary objective is to align OHRC/TMC-2/IIRS scenes, validate correspondences with geometry, enforce spatially distributed matches, and refine alignment to sub-pixel precision. Super-resolution is not the primary product, and it is not executed by the default CLI pipeline.

## How to reference it
This code remains useful as historical context for the original Chandrayaan-2 workflow, the dataset-curation process, and the earlier ESRGAN experiments. It can be consulted for dataset preparation, evaluation comparisons, or future research on learned enhancement pipelines.

## Important note
The legacy path is intentionally not part of the active LUNAR-ALIGN execution flow. The default CLI and API are designed for registration.
