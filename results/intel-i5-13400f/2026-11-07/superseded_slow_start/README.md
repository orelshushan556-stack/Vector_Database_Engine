# Superseded: main-pass results from a slow start-up window

These files are unchanged outputs of the main pass. The drift probes at the start of that pass
(`sanity_start`, 12:39 UTC) ran about 2.2x slower than the same probes at its end (`sanity_end`,
20:24 UTC), on the same saved graph with identical recall. The suites that ran in or just after
that window (`before_after_fix`, `crash_recovery`, `write_path`) were therefore re-run in a
follow-up pass. The published numbers come from the re-run in the parent folder. See
`docs/DECISIONS.md`, item 16.

How the re-run compares (computed from the files):

- `crash_recovery`: identical outcome in both passes (500/500 cycles passed, 0 acknowledged
  inserts lost, 125/125 injected torn tails truncated); replay times within a few percent.
- `write_path`: inserts/s within about 6% of the re-run, so the slow window had little effect
  on this I/O-bound suite.
- `before_after_fix`: the "after" recall agrees (worst build at ef 64: 0.989 here, 0.998 in
  the re-run). The "before" worst build differs (0.394 here, 0.641 in the re-run) because the
  engine seeds its level generator from `std::random_device`, so each pass draws different
  graphs; recall does not depend on speed. Both are real measurements of the pre-fix code, and
  the summary reports both.
