---
title: Data Cleaning Agent
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Data Cleaning Agent OpenEnv

A REST API environment for evaluating AI agents on multi-step data cleaning tasks.

## Endpoints

- `GET /health` — Health check
- `POST /reset` — Start a new episode
- `POST /step` — Execute a cleaning action
- `GET /state` — Get current state
