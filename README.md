# LC48 machine-generation corpus: 568 recorded execution rows

Author: Ethan Wang. Version v0.1.0, 16 October 2026. Public machine-record release.

Repository: https://github.com/Xuan-0929/LC48-translation-diagnostics

Code: MIT. Constructed sources and machine records: CC BY 4.0. See LICENSE.md for the exact scope.

This field-selected corpus joins the192 original LC48 translations,256 later four-arm translations and120 numeral-form-control outputs. It contains constructed Chinese sources, exact native prompts/input IDs, unchanged generated English/output IDs, stopping flags and pinned model identifiers. It does not include participant answers, semantic grades, annotation/root rationales, human qualification or consent fields, profiles, cloud host/SSH details, billing/admission evidence, model weights or tokenizer files.

There are60 source identifiers but54 distinct source strings: the numeral control has12 form identifiers for six old texts and six digit variants. There are444 unique model/native-input sequences among568 recorded execution rows. Repeated executions remain separate:64 four-arm base outputs exactly anchor to old source-faithful outputs;60 Chinese-word numeral outputs anchor to prior native inputs/English/output tokens. These are not568 independent samples or568 new discoveries.

## Offline replay

From any extracted directory, run:

    python3 -I code/replay_machine_records.py --out replay-result.json

Only Python's standard library is needed. It checks all manifest bytes, request hashes, source/translation hashes, native token arrays and stopping flags; recounts stage/condition coverage and distinct-content inventories; and checks the64+60 repeated-execution anchors. It does not load model weights, regenerate GPU outputs, validate private admission/billing or supply semantic judgments. The expected result is expected/replay_summary.json.

## Optional native tokenizer check

With transformers==4.51.3 and the pinned locally available assets:

    python3 -I code/verify_native_tokenizers.py --qwen-path PATH_TO_QWEN --mistral-path PATH_TO_MISTRAL --out tokenizer-check.json

The check verifies cached asset hashes, all568 chat-template texts/input-ID sequences and both exact output decodes. It uses local_files_only=True and no GPU. External weights/tokenizers are not redistributed and retain their own terms.

## Code scope

code/original_inference_runner_v01.py is an unchanged copy of the original executed inference runner. It records the actual reference generation recipe but expects the historical source-release and compute-admission inputs; those private administrative and human-derived records are excluded here. It is not invoked by offline replay, and this release does not claim a standalone complete GPU regeneration command or independently revalidate execution. data/generation_recipe.json summarizes its scientific settings. No new reference runner or new model execution is manufactured.

## Relation to the manuscript

This supports inspection of actual generated wording, native-token identity, source-form contrasts and exact old-output recurrence. It does NOT reproduce the paper's primary R/A/X/U retention bounds: the human-derived semantic data required for that analysis are absent and require a separately established access route. The numerical likelihood and model-reader experiments are also outside this selected corpus. Existing held replay packages preserve those records separately. No single-corpus completeness claim is made.

Designer roles and semantic-scenario identifiers describe the constructed design; they are not participant qualifications or new human judgments. The digit variants carry no new human source qualification. No semantic category is assigned to any output in this package.

## Access and licence status

This repository is the author-authorized public release of the selected machine records. Code under code/ is licensed under MIT. Constructed sources, recorded machine translations and associated research metadata are licensed under CC BY 4.0 to the extent of the author's rights. External model and tokenizer assets are not included and retain their own terms. See LICENSE.md, LICENSE-MIT.txt and LICENSE-CC-BY-4.0.txt.

The original 17-file local candidate remains unchanged privately; the publication copy updates only release documentation, licensing and repository metadata. All files under code/, data/ and expected/ preserve the candidate bytes. MANIFEST.json covers every published file except itself. Private source-path lineage remains outside this repository.

There is no DOI or independently archived deposit. Public access to these machine records does not establish a human-data access route, resolve the study's ethics status or imply manuscript submission or acceptance. No participant records, semantic grades or manuscript are published here.
