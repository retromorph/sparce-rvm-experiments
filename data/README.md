# Data contracts

`raw/` and `processed/` are untracked. Keep dataset downloads out of Git.

For each pair store: `event_id`, `timestamp`, `api_family`, `package`, `package_version`, `request`, `incumbent`, `alternative`, `verified_label`, `verifier_id`, `verifier_result_hash`, `environment_digest`, `split_id`, `split_policy`, `provenance_url`, and `license`. Unknown values must be represented as unknown, not invented. No hidden labels may be passed to retrieval, scoring or generation at inference.

Use a predeclared chronological and API-family group split; reserve scope-mismatched, previously-correct and negative examples as regression controls. Record exact deduplication decisions and source hashes. Treat raw benchmark sources as potentially untrusted; do not run provided code outside isolated containers.
