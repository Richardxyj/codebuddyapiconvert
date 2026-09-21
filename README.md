# CodeBuddy API Convert

CodeBuddy API Convert turns an existing WorkBuddy / CodeBuddy subscription into OpenAI-compatible and Anthropic-compatible local APIs. It is designed for personal machines and for a small team server that needs to reuse multiple accounts safely.

## Features

- OpenAI Chat Completions API: `POST /v1/chat/completions`
- OpenAI Responses API: `POST /v1/responses`, tuned for Codex CLI and Codex Desktop-style clients
- Anthropic Messages API: `POST /v1/messages`, suitable for Claude Code / CC Switch
- Model list and health endpoints: `GET /v1/models`, `GET /health`
- Streaming SSE, tool calls, reasoning streams, image input, and long-context projection
- Optional local authentication with `--api-key`
- Optional management console for account-fleet reuse:
  - browser-based account authorization or desktop `.info` import
  - up to 100 accounts with round-robin or manual routing
  - automatic credential refresh, cooldown after auth/quota/rate failures
  - credit balance, daily check-in, and request metrics
  - revocable client API keys stored as SHA-256 digests

The converter is stateless in single-account mode. The optional management console stores only credentials, account metadata, key hashes, and request counters; it never stores prompts or responses.

## Requirements

- Python 3.8 or newer
- A valid WorkBuddy / CodeBuddy account or desktop login
- Dependencies: `fastapi`, `uvicorn`, `httpx`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m core.converter --desensitize --log converter.log
```

Windows users can also run `start-converter.bat`. Set `PYTHON` first if `python` is not the interpreter you want to use.

Check the service:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/v1/models
```

The service listens on `127.0.0.1` by default. Do not expose it directly to an untrusted network without adding authentication and TLS.

## Credential Discovery

Single-account mode reads an existing desktop login from:

- Windows: `%LOCALAPPDATA%\CodeBuddyExtension\Data\Public\auth\*.info`
- macOS: `~/Library/Application Support/CodeBuddyExtension/Data/Public/auth/*.info`
- Linux: `~/.local/share/CodeBuddyExtension/Data/Public/auth/*.info`

Set `CODEBUDDY_AUTH_DIR` to use another directory. Do not commit `*.info` files.

## Client Configuration

### Codex CLI / Responses Clients

Merge this into `~/.codex/config.toml`:

```toml
[model_providers.workbuddy]
name = "WorkBuddy (via CodeBuddy API Convert)"
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
env_key = "CODEBUDDY2OPENAI_KEY"

[profiles.workbuddy]
model = "glm-5.2"
model_provider = "workbuddy"
```

```bash
export CODEBUDDY2OPENAI_KEY=any-value   # or your --api-key value
codex --profile workbuddy "your task"
```

### OpenAI-Compatible Clients

Configure:

- Base URL: `http://127.0.0.1:8787/v1`
- API key: empty, or the value passed to `--api-key`
- Model: a model returned by `/v1/models`, for example `glm-5.2` or `deepseek-v4-pro`

### Claude Code / CC Switch

- Base URL: `http://127.0.0.1:8787/v1/messages`
- API key: empty, or the value passed to `--api-key`
- Model: a WorkBuddy model name such as `deepseek-v4-pro`

## Account Fleet Mode

For a shared server or multiple accounts, run the management console instead:

```bash
export ADMIN_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(36))")
export CODEBUDDY2OPENAI_KEY=sk-wb-$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export CODEBUDDY_AUTH_DIR="$PWD/auth"
export MANAGEMENT_DATA_DIR="$PWD/management"
python -m admin.server
```

Open `http://127.0.0.1:8787/admin/` and add accounts through browser authorization or `.info` import.

Fleet routing rules:

- `round_robin` is the default; `manual` pins new requests to one selected account.
- Paused, invalid, cooling, and exhausted accounts are skipped.
- Upstream `401/403` cools an account for 5 minutes; `402/429` cools it for 30 minutes.
- Requests are bound to one account for their full lifetime; a partially emitted stream is never replayed.
- Credential refresh uses a per-account lock so concurrent requests do not race token updates.

Keep the console to a single process. It is intentionally single-process so account selection and in-flight credential state stay consistent.

### One-Command Docker Deployment

```bash
bash deploy/one-click/deploy.sh
```

The script generates `ADMIN_KEY` and the initial client key, creates `.env`, builds the image, and starts the console on `127.0.0.1:8787`. For a public endpoint, place it behind an HTTPS reverse proxy; see `deploy/admin/nginx.conf.example`.

### Standalone Docker

Edit `deploy/standalone/docker-compose.yml` to mount your auth directory, then:

```bash
docker compose -f deploy/standalone/docker-compose.yml up -d --build
```

## Security Notes

- Use `--api-key` or the console's client keys for local authentication.
- Keep `auth/`, `management/`, `.env`, `*.info`, and log files out of version control.
- Log files can contain request payloads. Use `--log` only while diagnosing an issue, and delete logs when finished.
- Put a public management console behind HTTPS. Its admin cookie is marked `Secure`, so plain HTTP login is intentionally unsupported.
- Do not run the console with multiple Uvicorn workers.

## Development

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

Run the syntax checks:

```bash
python -m compileall core admin tests
node --check admin/static/app.js
```

## Project Structure

```text
core/                    Protocol conversion service
admin/                   Optional account-fleet console
deploy/                  Docker, compose, and Nginx examples
tests/                   Unit tests
codex-codebuddy.example.toml
service_run.py           Headless entry point with port guard
```

## Acknowledgements

This project evolved from the ideas in [HanHan666666/codebuddy2openai](https://github.com/HanHan666666/codebuddy2openai). The browser authorization flow was implemented independently, with reference to the flow used by Sliverkiss/cpa-plugin.

## Legal Notice

This project is not affiliated with Tencent, WorkBuddy, CodeBuddy, OpenAI, or Anthropic. Use it only with subscriptions you are authorized to use, in compliance with the applicable service terms. The project is provided for personal learning and research; use it at your own risk.

## License

[MIT](./LICENSE)
