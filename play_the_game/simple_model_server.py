#!/usr/bin/env python3
"""
Simple Model Server for Pac-Man AI

A lightweight Flask server that loads the model and provides OpenAI-compatible API.
This avoids vLLM threading issues by loading the model once on the main thread.

Usage:
    python simple_model_server.py [--model-path PATH] [--port PORT]
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from flask import Flask, request, jsonify
import torch

# Setup Python path for OpenEnv
def setup_openenv_path():
    cwd = Path.cwd()
    project_root = cwd
    for _ in range(4):
        if (project_root / "src").exists():
            break
        project_root = project_root.parent
    else:
        project_root = Path(__file__).parent.parent
        if not (project_root / "src").exists():
            raise FileNotFoundError("Could not locate OpenEnv src directory")
    
    openenv_src = project_root / "src"
    if str(openenv_src) not in sys.path:
        sys.path.insert(0, str(openenv_src))
    
    return project_root

PROJECT_ROOT = setup_openenv_path()

# Import model libraries
try:
    from unsloth import FastLanguageModel
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("❌ Unsloth not available")
    sys.exit(1)

# Global state
model_state = {
    'model': None,
    'tokenizer': None,
    'model_name': None,
}

app = Flask(__name__)


def load_model(model_path: str):
    """Load model on main thread."""
    print(f"🔄 Loading model from: {model_path}")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_path),
            max_seq_length=2048,
            load_in_4bit=False,
            dtype=torch.bfloat16,
        )
        FastLanguageModel.for_inference(model)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {model.device}")
        print(f"   Dtype: {model.dtype}")
        
        return model, tokenizer
    
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models (OpenAI compatible)."""
    return jsonify({
        'object': 'list',
        'data': [{
            'id': model_state['model_name'],
            'object': 'model',
            'created': 1234567890,
            'owned_by': 'local',
        }]
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint."""
    if model_state['model'] is None or model_state['tokenizer'] is None:
        return jsonify({
            'error': {
                'message': 'Model not loaded',
                'type': 'server_error',
                'code': 500
            }
        }), 500
    
    try:
        data = request.json
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 128)
        top_p = data.get('top_p', 0.9)
        
        # Apply chat template
        text = model_state['tokenizer'].apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = model_state['tokenizer'](
            text, 
            return_tensors="pt"
        ).to(model_state['model'].device)
        
        # Generate
        with torch.no_grad():
            outputs = model_state['model'].generate(
                **inputs,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
                pad_token_id=model_state['tokenizer'].pad_token_id,
                eos_token_id=model_state['tokenizer'].eos_token_id,
                do_sample=True,
            )
        
        # Decode
        generated = model_state['tokenizer'].decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Return OpenAI-compatible response
        return jsonify({
            'id': 'chatcmpl-local',
            'object': 'chat.completion',
            'created': 1234567890,
            'model': model_state['model_name'],
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': generated,
                },
                'finish_reason': 'stop',
            }],
            'usage': {
                'prompt_tokens': inputs['input_ids'].shape[1],
                'completion_tokens': outputs.shape[1] - inputs['input_ids'].shape[1],
                'total_tokens': outputs.shape[1],
            }
        })
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': {
                'message': str(e),
                'type': 'server_error',
                'code': 500
            }
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_state['model'] is not None,
        'model_name': model_state['model_name'],
    })


def main():
    parser = argparse.ArgumentParser(description="Simple Model Server for Pac-Man AI")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./merged_model_old",
        help="Path to the model checkpoint (default: ./merged_model)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9810,
        help="Port to run the server on (default: 9810)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("🚀 SIMPLE MODEL SERVER FOR PAC-MAN AI")
    print("=" * 70)
    print(f"📦 Model: {model_path}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print(f"📡 OpenAI API endpoint: http://localhost:{args.port}/v1")
    print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    
    # Load model on main thread
    model, tokenizer = load_model(str(model_path))
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    model_state['model'] = model
    model_state['tokenizer'] = tokenizer
    model_state['model_name'] = model_path.name
    
    print("\n✅ Model loaded! Starting Flask server...")
    print(f"   Access API at: http://localhost:{args.port}/v1/chat/completions")
    print(f"   Health check: http://localhost:{args.port}/health")
    print("=" * 70)
    
    # Run Flask server
    app.run(
        host=args.host,
        port=args.port,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
