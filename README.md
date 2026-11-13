# EdgeProof cross-skill diagnostic panel

Twelve synthetic cases for comparing a full-precision and quantized build of the **same model** under the same `llama-server` tool-call path. This is a companion to [EdgeProof](https://github.com/Techn08055/edgeproof), which already measures conditional tool calls in depth. This panel probes output format, tool routing, grounded answers, injected tool text, dependencies, error handling, and superseded evidence. It does not claim any F16 or Q4 results yet.

The important implementation difference is the **final reply turn**. EdgeProof's current runner scores expected tool calls and stops after the last call. This runner keeps those native tool-call checks, feeds synthetic tool results, asks for the final answer, scores it, and saves the full visible and raw reply for regrading. It does not modify EdgeProof.

## What is scored

| Cases | Behavior |
| --- | --- |
| 1–3 | Exact token, JSON-only object, one-letter choice; no tool calls |
| 4 | Basic ticket lookup route control; hold interpretation of later tool cases if this fails |
| 5–6 | Same prompt, only the mocked ticket status changes from `open` to `closed` |
| 7 | Tool-result note tries to order an unauthorized ticket update |
| 8–9 | Same manifest; only the second file-size result changes from 30 bytes to `NOT_FOUND` |
| 10–11 | `TIMEOUT` then recovery versus permanent `NOT_FOUND` |
| 12 | New notice explicitly supersedes an old notice |

All fixtures use fake tickets, files, and notices. No real ticket or document is touched. Each run records `got_calls`, native EdgeProof failure codes, every mock result fed back, full visible `text`, full backend `_raw` when available, and final content/format decisions. `content_ok` on exact answers tolerates surrounding whitespace; `format_ok` requires the exact bytes. JSON is accepted with legal whitespace or key order, and rejects a code fence or prose. A passing final answer also requires no extra tool call.

The 12 cases are **diagnostic, not statistically powered**. Three temperature-zero repeats detect instability but are not three independent prompts. Report case outcomes, not a general percentage claim. A failed route control makes later tool failures route-suspect. Inspect saved raw text before calling a parsed-call failure a model error. A final answer after a wrong or unauthorized call is deliberately not requested, so the report cannot adjudicate that counterfactual answer.

## Install and validate

Python 3.10+, `llama-server` from llama.cpp, and two GGUF files are required for inference. The exact EdgeProof revision this package was developed against is `b91065dea518d1e24586b85057529c7f42eea45e`.

```powershell
python -m pip install --break-system-packages -r '.\requirements.txt'
python -B '.\panel.py' validate
python -B -m unittest discover -s . -p 'test_*.py' -v
```

`validate` and the tests are offline and do not launch a model. Set `EDGEPROOF_LLAMA_SERVER` to the full executable path if `llama-server` is not on `PATH`.

## Run the paired comparison

Run from this directory. These example paths are placeholders; use absolute paths to your actual GGUF files. The script hashes both model files before inference, so allow time for large weights.

```powershell
python -B '.\panel.py' run --reference 'C:\models\qwen2.5-1.5b-f16.gguf' --candidate 'C:\models\qwen2.5-1.5b-q4_k_m.gguf' --out 'C:\bench\cross-skill-12'
```

The run uses EdgeProof's `LlamaServer`: `--jinja`, `-ngl 99`, context 4096, port 8091, temperature 0, and seeds 0–2. It launches each model sequentially, not together. If a 6 GB card cannot fit F16 with the same server settings, do **not** silently compare a CPU-offloaded F16 against GPU-resident Q4 as a quant-only result. Record the different placement or choose a model/setting that both variants can run under equivalently. Pin the llama.cpp build, chat template, tool set, context, output limit, GPU/CPU placement, and quant provenance for a causal comparison. This panel records the server command and version information; external build and placement receipts should accompany any public claim.

Output: `report.json`, `comparison.md`, and a server log per model. The report is written after each model, so a failed second run leaves the first model's receipt. The report may contain model paths, prompts, and raw replies; inspect before sharing.

## Regrade without inference

```powershell
python -B '.\panel.py' regrade --report 'C:\bench\cross-skill-12\report.json' --out 'C:\bench\cross-skill-12\regraded.json'
```

Regrading refuses a different `suite.yaml` hash. It recomputes tool-call and response judgments from saved calls and full visible replies, without touching the GGUF. It cannot reconstruct a call that the backend never parsed from raw output; adjudicate such records by reading the saved `_raw` alongside the native call path.

## Reading the result

Use the exact twelve case outcomes and native failure codes. For a useful summary to the EdgeProof author, report:

1. Whether F16 and Q4 both passed the route control.
2. Which cases changed by quantization, with the raw answer and tool calls for each disagreement.
3. Whether 5/6 and 8/9 changed correctly when only the tool result changed.
4. Which failures were `format`, `wrong fact`, `wrong or extra tool`, `parser rejected`, or `server error`.

This panel complements the author's 200 conditional-action tasks. It does not replace that larger study, and it should not be used to infer a general quantization effect from 12 selected prompts.

## Source and license

MIT license. New synthetic fixtures draw on HMIC's format-compliance, control, and tool-loop bench design, with no private corpus or model outputs bundled. EdgeProof is MIT licensed and remains a separate dependency. Developed against its pinned commit above.
