#!/usr/bin/env python3
import os
import sys
import base64
import math
import time
import argparse
from PIL import Image
import tiktoken

# Try importing OpenAI and Anthropic, print warning if missing
try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

try:
    import anthropic
    HAS_ANTHROPIC_SDK = True
except ImportError:
    HAS_ANTHROPIC_SDK = False

# PRICING (USD per 1,000,000 tokens)
PRICING = {
    "openai": {
        "text_input": 2.50,
        "vision_input": 2.50,
    },
    "anthropic": {
        "text_input": 3.00,
        "vision_input": 3.00,
    }
}

def calculate_openai_vision_tokens(width: int, height: int, detail: str = "high") -> int:
    """
    Calculates GPT-4o vision tokens for an image of size width x height.
    
    Low detail is always 85 tokens.
    High detail resizes the image such that:
      - Fits within a 2048 x 2048 square.
      - Shortest side is 768px.
      - Divides the image into 512x512 tiles, costing 170 tokens each.
      - Plus a base overhead of 85 tokens.
    """
    if detail == "low":
        return 85
        
    w, h = width, height
    
    # Step 1: Scale down to fit 2048x2048
    if w > 2048 or h > 2048:
        scale = min(2048 / w, 2048 / h)
        w = int(w * scale)
        h = int(h * scale)
        
    # Step 2: Scale such that shortest side is 768px
    if w < h:
        scale = 768 / w
        w = 768
        h = int(h * scale)
    else:
        scale = 768 / h
        h = 768
        w = int(w * scale)
        
    # Step 3: Calculate tiles
    tiles_x = math.ceil(w / 512)
    tiles_y = math.ceil(h / 512)
    total_tiles = tiles_x * tiles_y
    
    # Step 4: Cost is 85 + 170 * tiles
    return 85 + 170 * total_tiles

def estimate_anthropic_vision_tokens(width: int, height: int) -> int:
    """
    Estimates Claude 3.5 Sonnet vision tokens using the 28x28 patch formula:
    Visual Tokens = ceil(width / 28) * ceil(height / 28)
    """
    # Note: Anthropic models may scale the image first if it's too large,
    # but for pages of ~1000px, this formula is highly accurate.
    return math.ceil(width / 28) * math.ceil(height / 28)

def count_local_text_tokens(text: str, model: str = "gpt-4o") -> int:
    """Counts text tokens locally using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base if model not recognized
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_base64_image(image_path: str) -> str:
    """Encodes an image file as base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def run_openai_evaluation(image_paths, prompt, model="gpt-4o"):
    """Runs GPT-4o evaluation with image inputs."""
    if not HAS_OPENAI_SDK:
        print("ERROR: OpenAI SDK not installed.")
        return None, None
        
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY not found. Skipping OpenAI API call.")
        return None, None
        
    print(f"\n--- Running OpenAI {model} Evaluation ---")
    client = OpenAI(api_key=api_key)
    
    # Build payload
    content = []
    for path in image_paths:
        base64_img = get_base64_image(path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_img}",
                "detail": "high"
            }
        })
    content.append({
        "type": "text",
        "text": prompt
    })
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
            temperature=0.0
        )
        latency = time.time() - start_time
        return response, latency
    except Exception as e:
        print(f"Error during OpenAI API execution: {e}")
        return None, None

def run_anthropic_evaluation(image_paths, prompt, model="claude-3-5-sonnet-20241022"):
    """Runs Claude 3.5 Sonnet evaluation with image inputs."""
    if not HAS_ANTHROPIC_SDK:
        print("ERROR: Anthropic SDK not installed.")
        return None, None, None
        
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("WARNING: ANTHROPIC_API_KEY not found. Skipping Anthropic API call.")
        return None, None, None
        
    print(f"\n--- Running Anthropic {model} Evaluation ---")
    client = anthropic.Anthropic(api_key=api_key)
    
    # Build payload
    content = []
    for path in image_paths:
        base64_img = get_base64_image(path)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_img
            }
        })
    content.append({
        "type": "text",
        "text": prompt
    })
    
    # We can also call the token counting API if supported in this version
    actual_tokens = None
    try:
        # Check if beta count_tokens is available
        if hasattr(client, "beta") and hasattr(client.beta, "messages") and hasattr(client.beta.messages, "count_tokens"):
            token_count_resp = client.beta.messages.count_tokens(
                model=model,
                messages=[{"role": "user", "content": content}]
            )
            actual_tokens = token_count_resp.input_tokens
            print(f"Anthropic count_tokens API reports total payload tokens: {actual_tokens}")
    except Exception as e:
        # Fallback if endpoint fails or isn't accessible
        pass
        
    start_time = time.time()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": content}]
        )
        latency = time.time() - start_time
        return response, latency, actual_tokens
    except Exception as e:
        print(f"Error during Anthropic API execution: {e}")
        return None, None, None

def print_cost_comparison(raw_text, image_paths):
    """Prints a comparative table of token usage and costs."""
    print("\n" + "="*80)
    print("                      UCO COST & TOKEN EFFICIENCY ANALYSIS")
    print("="*80)
    
    # Count local text tokens
    openai_text_tokens = count_local_text_tokens(raw_text, "gpt-4o")
    # For Claude, we estimate text tokens using OpenAI's tokenizer as a reasonable proxy
    # (Claude uses a similar byte-pair encoding tokenizer)
    claude_text_tokens = openai_text_tokens
    
    # Compute image dimensions and vision tokens
    total_openai_vision_tokens = 0
    total_claude_vision_tokens = 0
    
    print(f"{'Page / Image File':<25} | {'Dimensions':<12} | {'OpenAI Vision (T)':<18} | {'Claude Vision (T)':<18}")
    print("-" * 80)
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
            oa_t = calculate_openai_vision_tokens(w, h, "high")
            cl_t = estimate_anthropic_vision_tokens(w, h)
            total_openai_vision_tokens += oa_t
            total_claude_vision_tokens += cl_t
            print(f"{os.path.basename(path):<25} | {f'{w}x{h}':<12} | {oa_t:<18} | {cl_t:<18}")
        except Exception as e:
            print(f"Error reading image {path}: {e}")
            
    print("-" * 80)
    print(f"{'TOTAL VISION TOKENS':<25} | {'':<12} | {total_openai_vision_tokens:<18} | {total_claude_vision_tokens:<18}")
    print(f"{'RAW TEXT TOKENS':<25} | {'':<12} | {openai_text_tokens:<18} | {claude_text_tokens:<18}")
    print("="*80)
    
    # Calculate costs (per million tokens)
    # Price = (tokens / 1,000,000) * Price_per_M
    oa_text_cost = (openai_text_tokens / 1_000_000.0) * PRICING["openai"]["text_input"]
    oa_vision_cost = (total_openai_vision_tokens / 1_000_000.0) * PRICING["openai"]["vision_input"]
    
    cl_text_cost = (claude_text_tokens / 1_000_000.0) * PRICING["anthropic"]["text_input"]
    cl_vision_cost = (total_claude_vision_tokens / 1_000_000.0) * PRICING["anthropic"]["vision_input"]
    
    # Savings math
    oa_diff = oa_text_cost - oa_vision_cost
    oa_saving_pct = (oa_diff / oa_text_cost) * 100 if oa_text_cost > 0 else 0
    
    cl_diff = cl_text_cost - cl_vision_cost
    cl_saving_pct = (cl_diff / cl_text_cost) * 100 if cl_text_cost > 0 else 0
    
    print("COST COMPARISON (USD per 1,000 runs):")
    print(f"OpenAI GPT-4o:")
    print(f"  - As Raw Text:   ${oa_text_cost * 1000:.4f}")
    print(f"  - As UCO Image:  ${oa_vision_cost * 1000:.4f}")
    if oa_diff >= 0:
        print(f"  - SAVINGS:       ${oa_diff * 1000:.4f} ({oa_saving_pct:.1f}% cheaper)")
    else:
        print(f"  - PENALTY:       ${-oa_diff * 1000:.4f} ({-oa_saving_pct:.1f}% more expensive)")
        
    print(f"Anthropic Claude 3.5 Sonnet:")
    print(f"  - As Raw Text:   ${cl_text_cost * 1000:.4f}")
    print(f"  - As UCO Image:  ${cl_vision_cost * 1000:.4f}")
    if cl_diff >= 0:
        print(f"  - SAVINGS:       ${cl_diff * 1000:.4f} ({cl_saving_pct:.1f}% cheaper)")
    else:
        print(f"  - PENALTY:       ${-cl_diff * 1000:.4f} ({-cl_saving_pct:.1f}% more expensive)")
        
    print("="*80)
    print("Note: The above estimates exclude the small system instructions / user prompt text tokens.")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR accuracy and compute UCO token/cost savings.")
    parser.add_argument("raw_file", help="Path to original raw source code file (e.g. test_input.py).")
    parser.add_argument("images", nargs="+", help="One or more rendered PNG image paths.")
    parser.add_argument("--provider", choices=["openai", "anthropic", "both"], default="both", help="API provider to run (default: both).")
    parser.add_argument("--prompt", default="Analyze the code in the image carefully. Find the logic flaw that causes an infinite loop. Point out the exact line number, explain why it happens, and write the corrected version of the function.", help="Prompt to send to the model.")
    parser.add_argument("--openai-model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o).")
    parser.add_argument("--anthropic-model", default="claude-3-5-sonnet-20241022", help="Anthropic model to use (default: claude-3-5-sonnet-20241022).")
    
    args = parser.parse_args()
    
    # 1. Verify existence of files
    if not os.path.exists(args.raw_file):
        print(f"Error: Raw file {args.raw_file} not found.")
        sys.exit(1)
        
    for img_path in args.images:
        if not os.path.exists(img_path):
            print(f"Error: Rendered image {img_path} not found.")
            sys.exit(1)
            
    with open(args.raw_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
        
    # 2. Print Token and Cost comparison
    print_cost_comparison(raw_text, args.images)
    
    # Check SDK installations
    if args.provider in ["openai", "both"] and not HAS_OPENAI_SDK:
        print("Note: Install openai SDK to run OpenAI queries: .venv/bin/pip install openai")
    if args.provider in ["anthropic", "both"] and not HAS_ANTHROPIC_SDK:
        print("Note: Install anthropic SDK to run Anthropic queries: .venv/bin/pip install anthropic")
        
    # 3. Execute OpenAI request if requested and configured
    if args.provider in ["openai", "both"]:
        response, latency = run_openai_evaluation(args.images, args.prompt, args.openai_model)
        if response:
            print(f"Response Latency: {latency:.2f} seconds")
            print("Response:")
            print(response.choices[0].message.content)
            
    # 4. Execute Anthropic request if requested and configured
    if args.provider in ["anthropic", "both"]:
        response, latency, actual_tokens = run_anthropic_evaluation(args.images, args.prompt, args.anthropic_model)
        if response:
            print(f"Response Latency: {latency:.2f} seconds")
            print("Response:")
            print(response.content[0].text)
        
if __name__ == "__main__":
    # Fix potential argparse attribute name issue
    main()
