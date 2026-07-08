#!/usr/bin/env python3
import os
import sys
import math
import argparse
from PIL import Image, ImageDraw, ImageFont

# Standard macOS monospaced font search paths (in order of preference)
FONT_FALLBACKS = [
    "/System/Library/Fonts/Monaco.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/Library/Fonts/Courier New.ttf",
]

def find_monospace_font():
    """Attempts to find a monospace font on the system, falling back to default."""
    for path in FONT_FALLBACKS:
        if os.path.exists(path):
            return path
    return None

def get_font_metrics(font, sample_text="A"):
    """
    Measures character width and calculates line height for the loaded font.
    
    Using font.getlength() for character width provides the horizontal advance.
    Using font.getmetrics() provides the vertical metrics (ascent and descent).
    """
    try:
        # Get character advance width (horizontal advance)
        char_width = font.getlength(sample_text)
        # Fallback/validation: check bounding box if getlength returns 0
        if char_width <= 0:
            bbox = font.getbbox(sample_text)
            char_width = bbox[2] - bbox[0]
            
        # Get font vertical metrics
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        return char_width, line_height
    except AttributeError:
        # Fallback for default bitmap font which doesn't support getlength or getmetrics
        # Default font is typically 6x11 or similar
        return 6, 11

def render_text_to_images(
    text,
    font_path=None,
    font_size=14,
    line_spacing=1.25,
    margin=20,
    max_lines_per_page=80,
    tab_width=4,
    output_prefix="output_page",
    no_pagination=False
):
    """
    Renders text to one or more sharp, non-anti-aliased monochrome PNG images.
    """
    # 1. Process text lines and expand tabs
    lines = [line.expandtabs(tab_width) for line in text.splitlines()]
    if not lines:
        lines = [""]
        
    # 2. Font Loading
    if font_path and os.path.exists(font_path):
        selected_font_path = font_path
        print(f"Using user-specified font: {selected_font_path}")
    else:
        selected_font_path = find_monospace_font()
        if selected_font_path:
            print(f"Detected system monospace font: {selected_font_path}")
        else:
            print("WARNING: No monospace system font found. Falling back to default Pillow font.")
            print("Note: Default Pillow font size is fixed and cannot be scaled.")
            
    if selected_font_path:
        font = ImageFont.truetype(selected_font_path, font_size)
    else:
        font = ImageFont.load_default()
        
    # 3. Calculate character dimensions and layout metrics
    char_width, font_line_height = get_font_metrics(font)
    line_height = int(font_line_height * line_spacing)
    print(f"Metrics - Character Width: {char_width:.2f}px, Line Height (with spacing): {line_height}px")
    
    # 4. Handle pagination
    if no_pagination:
        pages = [lines]
    else:
        pages = [lines[i:i + max_lines_per_page] for i in range(0, len(lines), max_lines_per_page)]
        
    print(f"Total lines: {len(lines)}. Splitting into {len(pages)} page(s) (max {max_lines_per_page} lines/page).")
    
    generated_files = []
    
    # 5. Render pages
    for page_idx, page_lines in enumerate(pages, 1):
        # Calculate maximum line length in this page
        max_line_len = max(len(line) for line in page_lines) if page_lines else 0
        
        # Calculate dimensions
        img_width = int(max_line_len * char_width) + (margin * 2)
        img_height = int(len(page_lines) * line_height) + (margin * 2)
        
        # Ensure we have a minimum image size
        img_width = max(img_width, 100)
        img_height = max(img_height, 100)
        
        # Create grayscale (mode L) image with white background (255)
        # Note: L mode is better than 1-bit mode for compatibility with LLM APIs,
        # but setting fontmode to "1" keeps it strictly black-and-white (sharp edges)
        image = Image.new("L", (img_width, img_height), 255)
        draw = ImageDraw.Draw(image)
        
        # CRITICAL: Disable anti-aliasing by forcing fontmode to 1-bit monochrome
        draw.fontmode = "1"
        
        # Draw the text line-by-line
        for line_idx, line_text in enumerate(page_lines):
            x = margin
            y = margin + (line_idx * line_height)
            draw.text((x, y), line_text, fill=0, font=font)
            
        # Define output filename
        if len(pages) == 1 and output_prefix == "output_page":
            filename = "output.png"
        else:
            filename = f"{output_prefix}_{page_idx}.png"
            
        # Save image as lossless PNG with compression optimization
        image.save(filename, "PNG", optimize=True)
        print(f"Saved: {filename} ({img_width}x{img_height})")
        generated_files.append(filename)
        
    return generated_files

def main():
    parser = argparse.ArgumentParser(description="Render code/text into sharp, non-anti-aliased monochrome PNGs.")
    parser.add_argument("input_file", help="Path to the input text or code file.")
    parser.add_argument("-o", "--output-prefix", default="output_page", help="Prefix for the output PNG files.")
    parser.add_argument("-s", "--font-size", type=int, default=14, help="Font size in points (default: 14).")
    parser.add_argument("-l", "--line-spacing", type=float, default=1.25, help="Line spacing factor (default: 1.25).")
    parser.add_argument("-m", "--margin", type=int, default=20, help="Margin around the text in pixels (default: 20).")
    parser.add_argument("--max-lines", type=int, default=80, help="Max lines per page/image (default: 80).")
    parser.add_argument("--font-path", default=None, help="Explicit path to a TTF font file.")
    parser.add_argument("--no-pagination", action="store_true", help="Render everything as a single large PNG.")
    parser.add_argument("-t", "--tab-width", type=int, default=4, help="Number of spaces to expand tabs to (default: 4).")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist.")
        sys.exit(1)
        
    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    render_text_to_images(
        text=text,
        font_path=args.font_path,
        font_size=args.font_size,
        line_spacing=args.line_spacing,
        margin=args.margin,
        max_lines_per_page=args.max_lines,
        tab_width=args.tab_width,
        output_prefix=args.output_prefix,
        no_pagination=args.no_pagination
    )

if __name__ == "__main__":
    main()
