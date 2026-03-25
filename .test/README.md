Run the MDLM math render check with:

```bash
python3 .test/check_mdlm_math_render.py
```

The script builds `slides/mdllm.md` with the current `mkslides.yml` into `/tmp`, serves the generated site locally, renders it in headless Chrome, and fails if rendered slides still contain raw TeX delimiters or missing math output.

It expects a local Chrome/Chromium binary and network access for the CDN-hosted Reveal/Math assets referenced by the current config.
