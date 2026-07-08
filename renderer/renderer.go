package renderer

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"strings"
)

// wrapText wraps text at word boundaries so that no line exceeds maxChars.
// It expands tabs to 4 spaces and returns the wrapped lines and the maximum line length.
func wrapText(text string, maxChars int) ([]string, int) {
	text = strings.ReplaceAll(text, "\t", "    ")
	var wrapped []string
	lines := strings.Split(text, "\n")
	maxLineLen := 0

	for _, line := range lines {
		if len(line) == 0 {
			wrapped = append(wrapped, "")
			continue
		}

		runes := []rune(line)
		for len(runes) > 0 {
			if len(runes) <= maxChars {
				wrapped = append(wrapped, string(runes))
				if len(runes) > maxLineLen {
					maxLineLen = len(runes)
				}
				break
			}

			// Look for space to wrap at word boundary
			wrapIdx := maxChars
			for i := maxChars; i > 0; i-- {
				if runes[i] == ' ' {
					wrapIdx = i
					break
				}
			}

			// Wrap at wrapIdx
			segment := runes[:wrapIdx]
			wrapped = append(wrapped, string(segment))
			if len(segment) > maxLineLen {
				maxLineLen = len(segment)
			}

			// Remainder
			runes = runes[wrapIdx:]
			// Strip leading space of remainder
			if len(runes) > 0 && runes[0] == ' ' {
				runes = runes[1:]
			}
		}
	}
	return wrapped, maxLineLen
}

// RenderTextToPNG renders input text into a lossless, sharp, non-anti-aliased
// monochrome PNG using our custom embedded 10x16 bitmap font engine.
func RenderTextToPNG(text string) ([]byte, error) {
	const (
		maxChars   = 80
		charWidth  = 10
		charHeight = 16
		margin     = 20
	)

	// 1. Wrap the text to fit within maxChars characters per line
	lines, maxLineLen := wrapText(text, maxChars)

	// 2. Compute dynamic image dimensions based on text length
	width := maxLineLen*charWidth + margin*2
	height := len(lines)*charHeight + margin*2

	// Safe minimum bounds
	if width < 100 {
		width = 100
	}
	if height < 100 {
		height = 100
	}

	// 3. Create Grayscale (mode L) image canvas
	img := image.NewGray(image.Rect(0, 0, width, height))
	// Fill with white (255)
	draw.Draw(img, img.Bounds(), &image.Uniform{color.Gray{Y: 255}}, image.Point{}, draw.Src)

	// 4. Draw text character by character
	for lineIdx, line := range lines {
		yCellStart := margin + lineIdx*charHeight
		runes := []rune(line)
		for charIdx, char := range runes {
			xCellStart := margin + charIdx*charWidth

			// Map character code to ASCII range
			code := int(char)
			if code < 0 || code >= 128 {
				code = 63 // Fallback to '?' for characters outside ASCII 127
			}

			// Retrieve 16-row glyph bitmap
			glyph := FontData[code]
			for row := 0; row < charHeight; row++ {
				rowByte := glyph[row]
				for col := 0; col < 8; col++ {
					pixelOn := (rowByte >> (7 - col)) & 1
					if pixelOn == 1 {
						// Draw black pixel (0)
						// Add 1-pixel horizontal offset on the left to center the 8px glyph in the 10px cell
						img.SetGray(xCellStart+1+col, yCellStart+row, color.Gray{Y: 0})
					}
				}
			}
		}
	}

	// 5. Encode to lossless PNG using BestSpeed configuration
	var buf bytes.Buffer
	encoder := png.Encoder{
		CompressionLevel: png.BestSpeed,
	}
	err := encoder.Encode(&buf, img)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}
