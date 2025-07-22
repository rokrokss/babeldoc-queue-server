# BabelDOC Queue Backend
<sup>main-queue API server ＋ serverless worker</sup>

This repository contains two coordinated FastAPI services that together provide an **async, queue-backed PDF machine-translation pipeline powered by [BabelDOC](https://github.com/funstory-ai/BabelDOC)**.

| Component | Purpose | Default Port |
|-----------|---------|--------------|
| **Main Server** (`main_server.py`) | Handles file uploads, enqueueing, task tracking, download endpoint, and exposes a REST API to clients. | **8000** |
| **Worker Server** (`worker_server.py`) | Stateless worker that pulls a queued job, runs BabelDOC translation, uploads the result, and reports progress back. | **8001** |

> **License:** All source code in this repository is released under the **GNU Affero General Public License v3.0-or-later (AGPL-3.0)**.  
> See [`LICENSE`](./LICENSE) for the full text.
