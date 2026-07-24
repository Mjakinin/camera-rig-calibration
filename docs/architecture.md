# Architecture

## Execution graph

```text
capture/import
  → input preparation and intrinsics
  → raw marker detection/PnP
  → per-job observation_quality_v1
  → candidate analysis
  → one queue-wide review checkpoint
  → method jobs
  → method/common-support evaluation
  → comparison/report
```

The wizard only authors strict schema-v5 configs and one-dataset queues.
Interactive and automated runs use the same registries, preflight, and runtime
contracts.

Queue preflight prepares the dataset and detects raw observations once. It then
applies every job's immutable and configurable observation checks, analyzes
candidates on those accepted observations, validates every method, and writes
one report per job. `FAILED_PREFLIGHT` blocks only that method; independent
jobs may finish, but the queue is not published until the failed job is fixed,
rerun, or explicitly removed.

Legacy multi-dataset queues are partitioned into ordered one-dataset subqueues
when loaded.

## Scientific ownership

- AP01 owns its Root Camera and has no Reference Marker.
- AP02 owns its pose-graph Reference Marker.
- AP03 Single owns its diagnostic scale marker.
- AP03 Multi owns its primary robust marker set.
- Evaluation owns an independent post-method common anchor.

Selection uses observed connectivity and individually recorded measurements,
never intrinsics or project-specific camera names. Selection is lexicographic
over displayed measurements. The comparison fields, recommendation reasons,
and stable tie-breaks are persisted.

AP02 separately initializes and runs Static-only BA (diagnostic) and Combined
static/moving BA (primary). A Static-only runtime failure does not prevent the
independently initialized Combined BA. AP03 runs one COLMAP reconstruction and
then two scale stages; Multi is primary and Single is diagnostic.

## Configuration and queue snapshots

One queue contains one dataset. Common dataset, ArUco input, and evaluation
values are stored in `common`, while every entry points to a full independent
job snapshot. Queue-common values in a snapshot must match exactly. Duplicating
a row performs a deep copy.

Schema v1/v2/v3/v4 is migrated in memory to schema v5. Split AP03 Single/Multi rows
are merged only when their COLMAP, quality, input, and combined AP03 snapshots
are identical.

## Content identity and publication

Inputs have stable content hashes. Method fingerprints contain the complete
scientific snapshot and resolved selections. Execution identity is:

```text
(method fingerprint, input fingerprint)
```

```text
datasets/<category>/<factor-or-source>/<experiment>/
  inputs/<input-id>/
    raw_images/
    metadata/
    observations/<detection-id>/
      shared_*.csv
      connectivity_report.json
      debug_gallery/

results/<category>/<same-group>/<experiment>/
  methods/<method>/<variant>/executions/<input-id>/current/
  evaluations/
  comparisons/
  PUBLISHED.json
```

Capture, preparation, observations, jobs and evaluations begin below
`workspace/temporary_runs/<queue-id>/`. Publication builds complete dataset and
result `.incoming` snapshots, verifies content, and atomically swaps each
canonical experiment. `PUBLISHED.json` is the visibility boundary. A crash is
recovered from the transaction journal, and an incomplete queue never replaces
a known-good experiment.

Simulation factor folders are canonical. Only baseline-default aliases are
relative links (for example `fov/69.1deg → baseline/route2`); there is no
separate `_views` hierarchy.

## Reuse boundaries

```text
capture/import → preparation → raw detection/PnP → observation quality
→ COLMAP → method estimation/scale → evaluation
→ comparison
```

- Changing AP01 Root reruns AP01 estimation but can reuse COLMAP.
- Changing AP02 Reference Marker reruns AP02 graph initialization and BA.
- Changing AP03 scale markers reuses compatible COLMAP and reruns scale.
- Changing observation quality invalidates work from filtering onward.
- Changing the Evaluation Anchor reruns evaluation only.

AP01 and AP03 use distinct content-addressed COLMAP artifact families. A cache
key includes input and all COLMAP settings but excludes AP01 Root and AP03
scale-marker choices.

The active method implementations are explicit importable stages:

```text
AP01: observations → COLMAP → scale → candidates → solution → report
AP02: observations → graph → static/combined initialization
      → static/combined BA → report
AP03: observations → grouped COLMAP → inspection
      → single/multi scale → report
```

Each stage writes a schema-v5 `stage_manifest.json`. Resume skips completed
stages. The stage implementations do not alter transformation direction,
ArUco corner order, PnP, distortion, COLMAP conventions, residual definitions,
triangulation, or robust estimators.

## Runtime records

Adapters emit structured stage events around unchanged scientific runners.
Terminal output is flushed immediately and includes stage/job/queue elapsed
seconds, useful counts, and the log path. `timings.json` and
`run_manifest.json` preserve the same information. No estimated completion time
is fabricated.

AP02 optimizer reports include limits, actual `nfev`, success, status, message,
initial/final robust cost, reprojection metrics, loss, loss scale, and runtime.
Common evaluation reports supplied pose/support frame IDs and their
intersection; AP03 Single is additionally evaluated as a diagnostic.

## Legacy safety

Historical successful results are indexed through verified manifests and remain
byte-identical. Missing real inputs are labeled `input unavailable / not
rerunnable`. Incomplete-run cleanup never removes shared inputs or successful
results.
