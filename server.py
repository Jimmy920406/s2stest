import os
import json
import csv
import logging
import numpy as np
import httpx
import websockets
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# 啟動時載入 FAQ 資料
faq_answers = []
faq_embeddings = None

def load_faq():
    global faq_answers, faq_embeddings
    csv_path = os.path.join(os.path.dirname(__file__), "faq_rows.csv")
    embeddings = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            faq_answers.append(row["answer"])
            embeddings.append(json.loads(row["embedding"]))
    matrix = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    faq_embeddings = matrix / norms
    logger.info(f"[faq] 載入 {len(faq_answers)} 筆資料")

load_faq()

async def search_faq(query: str) -> str:
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={"model": "text-embedding-3-small", "input": query},
            timeout=10.0
        )
        query_vec = np.array(res.json()["data"][0]["embedding"], dtype=np.float32)

    query_vec = query_vec / np.linalg.norm(query_vec)
    similarities = faq_embeddings @ query_vec
    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])
    logger.info(f"[search] query={query}, similarity={best_score:.4f}")

    if best_score < 0.5:
        return "資料庫中找不到相關資訊。"
    return faq_answers[best_idx]

@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()

    gemini_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"

    async with websockets.connect(gemini_url) as gemini_ws:
        async def receive_from_client():
            try:
                while True:
                    data = await client_ws.receive_text()
                    await gemini_ws.send(data)
            except Exception as e:
                logger.error(f"[receive_from_client] error: {e}")

        async def receive_from_gemini():
            try:
                while True:
                    raw = await gemini_ws.recv()
                    if isinstance(raw, bytes):
                        response_text = raw.decode("utf-8")
                    else:
                        response_text = raw
                    response_data = json.loads(response_text)

                    if "toolCall" in response_data:
                        function_call = response_data["toolCall"]["functionCalls"][0]
                        call_id = function_call.get("id", "")
                        query = function_call["args"]["query"]
                        logger.info(f"[toolCall] query: {query}")

                        answer = await search_faq(query)

                        tool_response = {
                            "tool_response": {
                                "function_responses": [{
                                    "id": call_id,
                                    "name": "search_database",
                                    "response": {"output": answer}
                                }]
                            }
                        }
                        await gemini_ws.send(json.dumps(tool_response))

                    await client_ws.send_text(response_text)
            except Exception as e:
                logger.error(f"[receive_from_gemini] error: {e}")

        import asyncio
        await asyncio.gather(receive_from_client(), receive_from_gemini())
