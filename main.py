from fastapi import FastAPI, HTTPException
import google.generativeai as genai
from pydantic import BaseModel
import os
import logging
from typing import Optional, Dict, Any, List


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
api_key = os.getenv("GEMINI_API_KEY", "")

class GeminiApi:
    def __init__(self, model: str = "gemini-1.5-flash") -> None:
            self.model = model
            self._init_model()

    def _init_model(self) -> None:
        try:
            self.model_instance = genai.GenerativeModel(self.model)
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise
    async def text_conversation(self, conversation_history: str, prompt: str, new_message: str, audio_info: str = "") -> Optional[dict]:
        try:
            full_prompt = f"{prompt}\n\nConversation history:\n{conversation_history}\n\nNew message: {new_message}\n\nAudio info: {audio_info}"
            response = await self.model_instance.generate_content_async(full_prompt)
            ai_response = response.text
            return {"response": ai_response}
        except Exception as e:
            print(f"Error in text_conversation: {e}")
            return None

class GeminiRequest(BaseModel):
    prompt: str
    conversation_history: str
    new_message: str

gemini_api = GeminiApi()
@app.post("/")
async def return_request(request: GeminiRequest):
    try:
        result = await gemini_api.text_conversation(request.conversation_history, request.prompt, request.new_message)

        if result is None:
            raise HTTPException(status_code=503, detail="Failed to communicate with Gemini API after multiple attempts")

        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
