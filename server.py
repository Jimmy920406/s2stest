import os
import json
import httpx
import websockets
from fastapi import FastAPI, WebSocket
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
N8N_URL = os.getenv("N8N_WEBHOOK_URL")

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(client_ws: WebSocket):
    await client_ws.accept()
    
    gemini_url = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BiDiGenerateContent?key={GEMINI_API_KEY}"
    
    async with websockets.connect(gemini_url) as gemini_ws:
        async def receive_from_client():
            try:
                while True:
                    data = await client_ws.receive_text()
                    await gemini_ws.send(data)
            except Exception:
                pass

        async def receive_from_gemini():
            try:
                while True:
                    response_text = await gemini_ws.recv()
                    response_data = json.loads(response_text)
                    
                    if "toolCall" in response_data:
                        query = response_data["toolCall"]["functionCalls"][0]["args"]["query"]
                        
                        async with httpx.AsyncClient() as client:
                            n8n_res = await client.post(N8N_URL, json={"query": query})
                            n8n_data = n8n_res.json()
                            
                        tool_response = {
                            "tool_response": {
                                "function_responses": [{
                                    "name": "search_database",
                                    "response": {"output": n8n_data.get("answer", "")}
                                }]
                            }
                        }
                        await gemini_ws.send(json.dumps(tool_response))
                        
                    await client_ws.send_text(response_text)
            except Exception:
                pass

        import asyncio
        await asyncio.gather(receive_from_client(), receive_from_gemini())