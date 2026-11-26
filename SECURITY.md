# Security

## Reporting

Report vulnerabilities privately through GitHub Security Advisories ("Report a vulnerability" on
this repository). Do not open a public issue for a security problem.

## Scope

This repository is the engine: `oxe-server`, the `oxe` tool, `local-ai-memlat-boot`, the GGUF
loader, the vendored kernels and their FFI, and the release artifacts built from them. The `oxa`
coding agent, its sandbox, its provider credentials and its key store are part of
[oxstack/ox-llm](https://github.com/oxstack/ox-llm); report those there.

The server binds `127.0.0.1` by default, accepts only loopback `Host` names and refuses a
non-loopback bind without an API key file (`--api-key-file` or `OXE_API_KEY_FILE`). A model file,
a request or a client that breaks those guarantees is in scope.

## High severity

- A memory-safety bug in the C kernels or their FFI: a GGUF file, request or token batch that makes
  a kernel read or write outside its buffers, or a new `unsafe` block without bounds checks.
- A GGUF, tokenizer or ranking file that makes the loader map, read or write outside the file or
  its buffers.
- A request path that reaches the engine without the loopback `Host` check, without the configured
  API key, or outside the request-size and stream limits.
- Secret material in the repository or in a release artifact, and any step that prints or sends
  the configured API key.
- Qualcomm Hexagon/QAIRT SDK material in the tree: headers, IDL, stubs, skeletons, shared objects
  or ELF objects.
- A release artifact that does not match its `SHA256SUMS` or its build provenance.

## Not a vulnerability by design

- The server is for local clients and is bound to loopback unless an operator widens it with an
  API key; a local process with the same user can reach it.
- A model that produces wrong output is a correctness bug, not a security bug: see
  [CONTRIBUTING.md](CONTRIBUTING.md) for the lossless/reorder/lossy rules.
- Running a model or a build you downloaded and did not verify.

## Review guidelines

Read the change against the list above and report a finding only when it weakens one of those
boundaries; state the trigger and the impact and cite the changed lines. New `unsafe`/FFI,
process or network code without bounds and length checks is high severity. Size and integer
arithmetic on untrusted lengths (tensor shapes, token counts, offsets) is high severity when it
can wrap or truncate.

## Privacy

No weights, transcripts, home paths, user or host names, serials or keys in commits, issues or
release artifacts.
