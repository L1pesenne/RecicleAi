"""Chat local via Ollama. Nenhuma chave de API ou serviço externo."""
import os

import httpx
from fastapi import HTTPException

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.getenv("OLLAMA_MODEL", "hermes3:8b")
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "180"))


async def model_status():
    try:
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
        names = [item["name"] for item in response.json()["models"]]
        ready = CHAT_MODEL in names or f"{CHAT_MODEL}:latest" in names
        return {"provider": "ollama", "model": CHAT_MODEL,
                "available": ready, "reason": None if ready else "model_missing"}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return {"provider": "ollama", "model": CHAT_MODEL,
                "available": False, "reason": "service_unavailable"}


async def generate_reply(system_prompt, message, history):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": message})
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT, connect=5),
                                     trust_env=False) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model": CHAT_MODEL, "messages": messages, "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0.3, "num_ctx": 4096, "num_predict": 400},
            })
            response.raise_for_status()
        text = response.json()["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty response")
        return text.strip()
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "A IA local demorou para responder. Tente novamente.") from exc
    except httpx.HTTPStatusError as exc:
        detail = ("O modelo de conversa ainda não foi instalado neste computador."
                  if exc.response.status_code == 404 else
                  "A IA local não conseguiu concluir a resposta. Tente novamente.")
        raise HTTPException(503, detail) from exc
    except httpx.RequestError as exc:
        raise HTTPException(503, "A IA local está desligada. Inicie o Ollama e tente novamente.") from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(502, "A IA local retornou uma resposta inválida. Tente novamente.") from exc
