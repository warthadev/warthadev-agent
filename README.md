# warthadev-agent

Runtime agent internal untuk `warthadev-backend`. Service menerima task terautentikasi dari backend, lalu meneruskan request ke endpoint OpenAI-compatible 9Router.

## Jalankan

```bash
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
set -a
source .env
set +a
python agent_main.py
```

Health check:

```bash
curl http://127.0.0.1:8788/health
```

Endpoint utama hanya boleh dipanggil backend melalui header `X-Agent-Internal-Key`. API key model dikirim backend secara sementara dan tidak disimpan oleh service.

## Docker

```bash
docker build -t warthadev-agent .
docker run --env-file .env -p 127.0.0.1:8788:8788 warthadev-agent
```

Jalankan 9Router di endpoint `AGENT_ROUTER_BASE_URL`, lalu isi `AGENT_INTERNAL_API_KEY` yang sama di agent dan backend.
