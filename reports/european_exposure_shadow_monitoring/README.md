# European Exposure Shadow Monitoring

This directory records locally frozen research specifications for European
exposure challengers. It does not activate a production feature and it is not a
prospective timestamp witness until the spec is committed/pushed or anchored in
an independently immutable store.

## Registered arms

- `POSITIVE_BRIDGE_020`: formula frozen locally as a post-hoc shadow hypothesis.
- `EURO_RESIDUAL_NEFF_V0`: design hypothesis only; no numerical formula,
  artifact or prospective prediction is registered yet. Its structural design
  is recorded in `euro_residual_neff_v0_design.md`.

The hypothetical missing-season point schedule discussed on 2026-09-01 is not
a registered candidate.

Production and shadow predictions must never share a ledger. The current ledger
implementation keys revisions and settlement selection by `match_id`, so a
challenger row in the production chain would become a production revision. Each
frozen arm requires a separate JSONL hash chain and a separate external head
anchor.

`positive_bridge_020_shadow_spec.json` is executable and immutable once a run
references its SHA-256. Mutable run status, artifact paths and ledger heads must
be written to a separate run manifest or registry; editing those values into the
spec would silently change the candidate identity.

The detailed working plan is generated under
`output/european_exposure_shadow_monitoring_2026_27/SHADOW_MONITORING_PLAN.md`.
That ignored output is useful for operations but is not itself an evidence
anchor.

## Local AO-core start

`POSITIVE_BRIDGE_020` was started locally for the `AO_CORE_SHADOW` scope on
2026-09-02. Its 396-record dedicated ledger is
`data/prediction_ledger/shadow/positive_bridge_020/ao_core_shadow_2026_27.jsonl`
and its head hash is
`1aa79d429d7318e3de84f0ae1df5f9aac741179bd6099f6d37d823c7bb09eeb2`.
The candidate artifact manifest SHA-256 is
`87c1f939b3fba962e8f6f8ac803f032cbe3f2d77c8283b4d45fc9ad0127a18b7`.

This is an internally verifiable local start. No commit, push or independent
timestamp anchor was created, so it does not prove the creation time to an
external observer. Candidate-specific ML, Domestic Poisson and the final served
ensemble remain unbuilt.
