from flask import Flask, request, jsonify
from gemini import GeminiApi
import asyncio

app = Flask(__name__)

GEMINI_API_KEY = os.getenv("Api_key", "")
gemini_api = GeminiApi(api_key=GEMINI_API_KEY)

@app.route('/text_conversation', methods=['POST'])
async def text_conversation():
    data = request.json
    full_context = data.get('full_context')
    message = data.get('message')
    
    if not full_context or not message:
        return jsonify({'error': 'Missing full_context or message'}), 400

    try:
        response = await gemini_api.text_conversation(full_context, message)
        return jsonify(response)
    except Exception as e:
        print(f"Error in text_conversation: {e}")
        return jsonify({'error': str(e), 'response': "Sorry, I couldn't process that."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
