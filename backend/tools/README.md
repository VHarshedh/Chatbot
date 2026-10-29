# Scientific tool and retrieval upgrades

The agents receive compact inspection cards for explicitly named structure files
in `public`, `data`, `DFT`, and `Analysis`. These preserve file PBC and report frame
counts, composition, cell geometry, and an O-H-based phase estimate. Generated
scripts can import the analysis, DFT, MD, training, and SLURM helper modules from
`backend.tools`; the executor mounts these modules read-only in the container.
SLURM requires a configured cluster and is unavailable over the network-isolated
sandbox. Submission errors now raise instead of returning a fabricated job ID.

Scientific execution runs through AST validation, Docker, and output auditing.
Invalid syntax, known ASE tuple/API mistakes, headless plot calls, and explicit
X-Y/X-Z mismatches are checked before Docker starts. Output audits reject explicit
failure messages, missing requested plots, empty/corrupt artifacts, suspiciously
small requested raster plots, and boundary counts exceeding an inspected crystal
partition. `process_exit_code` preserves zero when an output audit rejects a run;
`exit_code=3` prevents that result from entering successful-execution memory.
Artifact paths remain compatible with the UI; `artifact_metadata` adds dimensions
and sizes. Both SDK and CLI paths preserve stdout and stderr independently and
accept multiple read-only inputs with distinct basenames.

Analysis is deterministic and data-dependent. Density is total atomic mass density
in g/cm3 along the cell-normal coordinate. RDF uses bulk periodic shell-volume
normalization and excludes self pairs. Survival uses continuous membership with
stable atom ordering. Hydroxyl fraction counts singly H-coordinated oxygen among
all oxygen atoms. The O-H phase partition includes hydroxyls and may misclassify
protonated lattice oxygen; boundary selection identifies the two outer crystal
z-layers, not a general enclosing three-dimensional surface. These are explicit
geometric heuristics, not a universal chemical classifier. Output auditing checks
observable evidence and image integrity, not visual semantics or scientific truth.

Training validation checks finite `.raw` arrays and atom/frame shapes. Splitting
writes disjoint `Train` and `Test` directories and preserves original input data;
use `Train` for fitting, never the unsplit parent. Existing split/set outputs are
rejected to prevent silent replacement. QE parsing reads actual `pw.out` files
with ASE and converts eV to Ry. Malformed output raises; absent output is skipped.

Retrieval retains cosine gating, BM25/RRF, and metadata reranking. Entity-linked
graph facts and notebook relationships replace substring-triggered material facts.
Explicit multipart queries are decomposed; executable sibling cells contribute
compact dependency outlines (not complete runnable scripts). Notebook re-indexing
removes obsolete cells. TiO2/IrO2 conflicts are excluded from reference retrieval
and memory. Notebook content is no longer prepended by the API independently;
explicit notebook requests enter the same retrieval query. XML blocks are escaped
and bounded conservatively to 1,800 UTF-8 bytes (stricter than 1,800 tokens), with
complete blocks omitted when they do not fit. Runtime inspection cards and tool
signatures are separate code-generation inputs. Local API diagnostics describe
host-installed packages; container versions can differ.

To refresh existing indexed source and notebook content after this upgrade:

```powershell
python -m backend.rag.vectorstore
python -m backend.rag.notebook_parser
```

Rebuild an existing sandbox image to include `pydantic-settings`, used by the
configured SLURM helper:

```powershell
docker build -t chatbox-sandbox:latest backend/sandbox
```

The regression suite includes real ASE structures, density mass conservation,
RDF peaks, continuous survival, disjoint training sets, static/output validation,
stream separation, multiple input mounts, and isolated Docker CLI fallback tests.
Focused runs use `--no-cov` because the repository's default 95% coverage requirement
applies to the entire backend, not a selected regression subset.
