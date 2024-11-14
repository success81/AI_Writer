from flask import Flask, render_template, request, jsonify, session
from datetime import datetime
import os
import vertexai
from vertexai.preview.generative_models import GenerativeModel

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Initialize Vertex AI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"
PROJECT_ID = "winter-cogency-436501-g9"
REGION = "us-central1"
vertexai.init(project=PROJECT_ID, location=REGION)

def call_gemini_flash(prompt, conversation_history=None):
    """Call Gemini Flash model with conversation history"""
    model = GenerativeModel("gemini-1.5-pro-002")
    
    # Build message history
    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({
                "role": "user" if msg["type"] == "user" else "assistant",
                "parts": [{"text": msg["content"]}]
            })
    
    # Add current prompt
    messages.append({"role": "user", "parts": [{"text": prompt}]})
    
    response = model.generate_content(
        contents=messages,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 8000,
            "top_p": 0.8,
            "top_k": 40
        }
    )
    return response.text

@app.route('/')
def index():
    # Initialize conversation history in session
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        selected_text = data.get('text', '').strip()
        context_before = data.get('contextBefore', '').strip()
        context_after = data.get('contextAfter', '').strip()
        user_request = data.get('question', '').strip()
        
        # Get conversation history from session
        conversation_history = session.get('conversation_history', [])
        
        # Build context-aware prompt
        prompt = f"""Current conversation context:
{build_conversation_summary(conversation_history)}

Document context:
BEFORE: {context_before}
SELECTED TEXT: {selected_text}
AFTER: {context_after}

Current request: {user_request}

Instructions:
1. Use the conversation history to maintain context
2. Provide direct responses without explanations
3. If this is a follow-up request, consider previous responses
4. Keep responses relevant to both the current selection and conversation flow
5. If generating replacements, ensure they flow with surrounding text

Remember: Be direct, contextual, and maintain narrative consistency."""

        # Log for debugging
        print(f"\nFull context and prompt:\n{prompt}")
        
        # Get response with conversation history
        response = call_gemini_flash(prompt, conversation_history)
        
        # Clean the response
        response = clean_response(response)
        
        # Update conversation history
        conversation_history.append({
            "type": "user",
            "content": user_request,
            "selection": selected_text,
            "timestamp": str(datetime.now())
        })
        conversation_history.append({
            "type": "assistant",
            "content": response,
            "timestamp": str(datetime.now())
        })
        
        # Keep only last 10 messages for context
        if len(conversation_history) > 10:
            conversation_history = conversation_history[-10:]
        
        session['conversation_history'] = conversation_history
        
        return jsonify({
            'response': response,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'An error occurred processing your request',
            'details': str(e),
            'status': 'error'
        }), 500

def build_conversation_summary(history):
    """Build a concise summary of the conversation history"""
    if not history:
        return "No previous conversation."
        
    summary = "Recent conversation:\n"
    for msg in history[-4:]:  # Only last 4 messages for immediate context
        prefix = "User" if msg["type"] == "user" else "Assistant"
        if msg["type"] == "user" and "selection" in msg:
            summary += f"{prefix} (Selected: '{msg['selection']}'): {msg['content']}\n"
        else:
            summary += f"{prefix}: {msg['content']}\n"
    return summary

def clean_response(response):
    """Clean up common AI verbosity patterns"""
    response = response.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "Here's", "I understand", "Let me", "I'll help",
        "Based on", "To answer", "I can", "Sure,", "Well,"
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            if response.startswith(','):
                response = response[1:].strip()
    
    return response

if __name__ == '__main__':
    app.run(debug=True)


