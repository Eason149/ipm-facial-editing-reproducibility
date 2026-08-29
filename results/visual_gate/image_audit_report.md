# Image Input and Condition Mapping Audit

## Decision

**PASS_INPUT_MAPPING_GATE**

The E-Prime logs provide a consistent, explicit mapping between image filename, identity, CondID, the four edit fields, Original, and IsBack. All 68 required physical facial stimuli were found uniquely. The input and mapping gate therefore passes, permitting calculation of image metrics.

## Evidence recovered

- E-Prime log files parsed: 30.
- Image-trial log records parsed: 14040.
- Unique referenced image filenames: 72.
- Mapping conflicts across logs: 0.
- Four target identities present in logs: F_1, F_2, M_1, M_2.
- Complete 16-condition factorial mapping for every identity: True.
- Original-image references present for every identity: True.
- Physical formal edited images found: 64/64.
- Physical original identity images found: 4/4.

## Condition coding

- FSlim is recorded as 0/1 and can be retained numerically.
- Skin is recorded as 0/1 and can be retained numerically.
- Eye is recorded as 1/2 and maps to 0/1 as 1->0 and 2->1.
- Mouth is recorded as 1/2 and maps to 0/1 as 1->0 and 2->1.
- Formal factorial trials use CondID 2-17 with Original=0 and IsBack=0.
- CondID 1 / filename index (1) is referenced as the original/control image and was not part of the formal 2x2x2x2 edited-condition analysis.

The numeric direction is recovered, but the verbal meaning and actual editing parameters of level 0 versus level 1 are not recoverable from the available images because those files are missing.

## Questions required by the audit

| Question | Finding |
| --- | --- |
| All four original images found? | Yes: the four experiment-referenced index-(1) files. |
| All 16 edited combinations found? | Yes, for each of the four identities (64 files). |
| Complete conditions for each identity? | Yes in E-Prime records and on disk. |
| Consistent resolution/color space/cropping/compression? | See `image_input_inventory.csv`; byte-level hashes and decoded metadata were recorded. |
| Image-to-behavior mapping? | Explicit in E-Prime logs via Picture, Stimtype and CondID. |
| Image-to-EEG mapping? | CondID and Stimtype are recoverable through epoch endCode. |
| Were originals in the formal factorial model? | No. Formal edited trials were CondID 2-17; originals were controls. |
| Coding direction restored? | Numeric coding yes; semantic interpretation will be checked against measured pixel/geometry changes. |

## Gate consequence

The audit authorizes the next stage (landmarks and G/A/I metrics). It does not itself establish construct validity or any behavioral/EEG result.
