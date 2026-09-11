# Legacy Repository Audit

## Scope
This audit records the original Chandrayaan-2 prototype code and distinguishes it from the new LUNAR-ALIGN architecture. The purpose is to preserve the valid data-ingestion and overlap-preprocessing logic while retiring the ESRGAN-first research path from the primary execution pipeline.

## Executive Summary
The original repository was built around the ISRO Inter-IIT Chandrayaan-2 problem: generate high-resolution lunar terrain from TMC-2 imagery using overlapping OHRC data, then create a lunar atlas. The project contains a useful data-processing pipeline for:

- discovering OHRC/TMC-2 overlaps,
- extracting metadata from XML files,
- converting `.img` data to compressed NumPy arrays,
- calculating spatial overlap and bounding boxes,
- applying affine/geometric preprocessing for common-region extraction.

These components are valid building blocks for a registration-oriented system. The repository also contains a substantial ESRGAN/super-resolution path, which is no longer the primary objective for LUNAR-ALIGN. That path should be moved to `legacy/` documentation status and not executed by the active registration pipeline.

---

## Original Component Audit

### 1. README and problem framing
| Field | Details |
|---|---|
| Original component | `README.md` |
| Purpose | Describes the original ISRO Inter-IIT problem, the dataset workflow, the super-resolution approach, the model training process, and atlas generation. |
| Input | Problem statement, dataset methodology, image pairs, model training notes. |
| Output | Design notes and documentation for ESRGAN-based enhancement and global map generation. |
| Dependencies | None beyond repo-local documentation. |
| Can reuse? | Yes, for historical context and migration rationale. |
| Where moved | Retained as project context; new README will replace it with LUNAR-ALIGN description. |
| What must be replaced? | Problem framing and technical objective. |

### 2. OHRC/TMC preprocessing notebooks
| Field | Details |
|---|---|
| Original component | `ohrc_tmc_preprocessing.ipynb`, `executable_1.ipynb`, `demonstration/executable_1.ipynb` |
| Purpose | XML parsing, `.img` to `.npz` conversion, overlap detection, corner-coordinate extraction, common-region cropping, affine preprocessing, and demo execution. |
| Input | Binary `.img` files, XML metadata, coordinate CSVs. |
| Output | Compressed `.npz` NumPy arrays, cropped common regions, aligned image patches. |
| Dependencies | `numpy`, `pandas`, `opencv-python`, `PIL`, `xml.etree.ElementTree`, `os` |
| Can reuse? | Yes, as the foundation for the new ingestion and preprocessing flow. |
| Where moved | Preserved conceptually in `core/ingestion/`, `core/metadata/`, `core/geometry/`, `core/preprocessing/`. |
| What must be replaced? | Demo- and notebook-specific execution flow; converted to modular CLI/service pipeline. |

### 3. Coordinate generation / overlap logic
| Field | Details |
|---|---|
| Original component | `coordinates_ohrc.csv`, `coordinates_tmc2.csv`, `new_map.py` |
| Purpose | Builds coordinate tables for image corners and performs geographic overlap comparison and map patch generation. |
| Input | Image filenames, corner coordinates, longitudes/latitudes, polygon membership tests. |
| Output | CSV datasets and candidate overlap pairs for map generation / patch extraction. |
| Dependencies | `pandas`, `numpy`, `geojson`, `turfpy`, `numba` |
| Can reuse? | Yes, in a refactored geometry engine. |
| Where moved | Replaced by `core/geometry/coordinates.py`, `haversine.py`, `overlap.py`. |
| What must be replaced? | Hardcoded map-generation logic and earlier atlas code path. |

### 4. XML metadata extraction
| Field | Details |
|---|---|
| Original component | Notebook metadata extraction cells in the preprocessing notebooks |
| Purpose | Parse XML metadata fields (image size, coordinate offset, pixel resolution, sensor details, geometry metadata). |
| Input | `.xml` sidecars for OHRC and TMC images. |
| Output | Structured metadata dictionaries / DataFrames. |
| Dependencies | `xml.etree.ElementTree`, `pandas` |
| Can reuse? | Yes, directly retained and formalized in `core/metadata/xml_parser.py` and `pds4_parser.py`. |
| Where moved | `core/metadata/` |
| What must be replaced? | Notebook ad hoc parsing and direct embedded logic. |

### 5. `.img` handling and `.npz` conversion
| Field | Details |
|---|---|
| Original component | Notebook binary readers for `.img` files |
| Purpose | Read ISRO binary image payloads and save compressed NumPy arrays. |
| Input | Raw `.img` files. |
| Output | `.npz` representation for reduced I/O. |
| Dependencies | `numpy`, `os`, file I/O |
| Can reuse? | Yes, highly relevant to ingestion. |
| Where moved | `core/ingestion/img_loader.py`, `npz_loader.py`, `dataset_builder.py` |
| What must be replaced? | Hardcoded paths and direct notebook logic. |

### 6. Common-region / overlap extraction
| Field | Details |
|---|---|
| Original component | Overlap crop and interpolation logic inside notebook cells |
| Purpose | Use OHRC/TMC corner coordinates to identify the overlapping region, compute bounding box, mask invalid pixels, and prepare aligned local patches. |
| Input | OHRC/TMC coordinates, geometric metadata, images. |
| Output | Cropped aligned common regions with masks. |
| Dependencies | `numpy`, `PIL`, `opencv`, `pandas` |
| Can reuse? | Yes, this is central to LUNAR-ALIGN ingestion and preprocessing. |
| Where moved | `core/geometry/overlap.py`, `core/preprocessing/masking.py`, `core/preprocessing/tiling.py`, `core/preprocessing/normalization.py` |
| What must be replaced? | Notebook-only processing sequence; convert to modular pipeline. |

### 7. Affine preprocessing and image alignment
| Field | Details |
|---|---|
| Original component | Notebook affine alignment logic |
| Purpose | Apply affine transforms to coarsely align OHRC and TMC images within the common region. |
| Input | Overlap bounding boxes, corner points, local image arrays. |
| Output | Standardized image pair and transform estimate. |
| Dependencies | `numpy`, `opencv` |
| Can reuse? | Yes, as initial geometric alignment stage. |
| Where moved | `core/geometry/affine.py`, `core/registration/transformation.py` |
| What must be replaced? | Hardcoded notebook assumptions and lack of validated object abstractions. |

### 8. Demonstration and trained model artifacts
| Field | Details |
|---|---|
| Original component | `demonstration/`, `train_model/main.py`, TensorFlow model files |
| Purpose | Demo ESRGAN network and inference flow; local saved model output. |
| Input | TMC images, OHRC pair data, model checkpoint / SavedModel. |
| Output | High-resolution reconstructed image. |
| Dependencies | TensorFlow, image processing, saved model format. |
| Can reuse? | Only historical reference; not in active pipeline. |
| Where moved | `legacy/` documentation and code references; the active system is now registration-focused. |
| What must be replaced? | Primary execution and objective: ESRGAN has been removed from the main pipeline. |

### 9. Map-generation code
| Field | Details |
|---|---|
| Original component | `new_map.py` |
| Purpose | Builds global atlas / patch extraction based on lat/long boxes and image coverage. |
| Input | `coordinates_tmc2.csv`, polygon geometry, image metadata. |
| Output | Candidate map tiles and global atlas patching logic. |
| Dependencies | `pandas`, `geojson`, `turfpy`, `numba` |
| Can reuse? | Partially; it is a geographic coverage tool but not required for the registration MVP. |
| Where moved | Geometry and coverage logic will be re-expressed in `core/spatial/` and `core/geometry/overlap.py`. |
| What must be replaced? | The atlas generation logic and its direct execution path. |

### 10. Legacy dependencies and hardcoded assumptions
| Field | Details |
|---|---|
| Original component | All notebooks and scripts |
| Purpose | Perform data processing in a notebook-first, path-specific workflow. |
| Input | Hardcoded local paths such as `ohrc/`, `tmc2/`, `demonstration/model`, etc. |
| Output | Local patch and model artifacts in the workspace. |
| Dependencies | Many legacy, notebook-specific imports and environment assumptions. |
| Can reuse? | Limited; only the underlying logic is valid, not the path assumptions. |
| Where moved | To modular code with configurable paths and YAML settings. |
| What must be replaced? | Hardcoded dataset layout, embedded filenames, and notebook execution order. |

---

## Reusable Legacy Building Blocks
The following components should be preserved and refactored into the new architecture:

1. OHRC/TMC metadata extraction from XML sidecars;
2. Binary `.img` to `.npz` conversion;
3. Corner-coordinate estimation and geographic overlap detection;
4. Haversine-based overlap logic;
5. Common-region identification and bounding-box extraction;
6. Initial affine/geometric alignment;
7. Patch extraction and masking of non-overlapping pixels;
8. Compression and data organization for lunar scene pairs.

These become the foundation of the `LunarImagePair` ingestion flow in `core/ingestion/` and related modules.

---

## Legacy Components That Must Be Retired from the Primary Pipeline
The following are no longer the goal of the active system:

1. ESRGAN-based super-resolution as the main execution objective;
2. TensorFlow training pipeline as the primary application flow;
3. Notebook-first execution as the main interface;
4. Atlas-generation as the main product; this becomes a downstream use case;
5. Hardcoded demonstration paths and local file assumptions;
6. Model outputs treated as the final product instead of registered imagery.

---

## Planned Migration
The repository will be transformed toward LUNAR-ALIGN with the following migration pattern:

- `README.md` and documentation rewritten to focus on registration research rather than super-resolution.
- Legacy ESRGAN code moved to `legacy/` with explanatory notes.
- Data ingestion refactored into `core/ingestion/`, `core/metadata/`, `core/geometry/`, and `core/preprocessing/`.
- Core matching and verification logic constructed around a common matcher interface and adaptive policy.
- CLI-first execution created via `scripts/run_registration.py`.
- FastAPI backend built after CLI succeeds.

This is a refactor preserving the scientific foundation while replacing the primary objective from super-resolution to robust multimodal lunar registration.
