# Data dictionary

UTF-8 JSON/JSONL; exact strings and code-point sequences are preserved. No whitespace/Unicode normalization is used for content identity.

- data/sources60.json: original48 constructed source identifiers plus12 numeral-form identifiers; source text/SHA256, lexical family, design structure/role and scenario ID. Variant original_source_id and numeral_form retain the construction relation.
- data/generations568.jsonl: one row per recorded stage/request, not one per unique source or outcome. record_id is stage:request_id. Original request/model/revision/source/prompt identifiers, native user/rendered text and IDs, exact output text and IDs and original stopping flags are preserved.
- data/model_assets.json: immutable model revisions and expected public asset hashes/file names. No asset bytes or local model directories are included.
- data/generation_recipe.json: original scientific recipe. It is not a new execution receipt.
- PROVENANCE.json: field allowlist, source-document hashes and exclusions. Absolute file-origin paths remain outside the candidate.
- expected/replay_summary.json: exact offline inventory and anchor readout, without language/fidelity judgments.

source_id is an identifier, not an independent unit. source_sha256 and translation_sha256 digest exact UTF-8 bytes. native_request_sha256 hashes the original eight scientific request fields in canonical JSON; stage and export-only fields are not injected into that digest. generated_token_N includes native EOS where present. ended_with_EOS is a stopping observation, not semantic completeness. generation_error=null means no recorded execution exception, not no translation error. No R/A/X/U or human review value exists in these records.
