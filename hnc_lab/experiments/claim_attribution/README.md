# Claim attribution experiment

This folder is reserved for the first manual claim attribution pass.

## Relation to the main HNC document

This experiment maps to:

- Section 8.1, information conservation.
- The source support path in `PROJECT_STATUS.md`.
- Project status Section 6.2, source accounting needs attribution.

## Planned measurements

The first implementation should take a small set of generated answers, split them into claims, and write `ClaimAttributionRow` plus `VerificationResult` records with:

- claim text,
- source label,
- evidence references,
- support score,
- verifier result,
- attribution confidence.

## Evidence boundary

Manual labels are useful for bootstrapping the rubric. They should be reported as small audit data, not as broad evidence until label rules and inter annotator checks exist.
