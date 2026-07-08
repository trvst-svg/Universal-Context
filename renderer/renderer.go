package renderer

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"strings"
	"sync"

	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/font/sfnt"
	"golang.org/x/image/math/fixed"
)

// Standard macOS monospaced font search paths (in order of preference)
var fontPaths = []string{
	"/System/Library/Fonts/Monaco.ttf",
	"/System/Library/Fonts/Menlo.ttc",
	"/System/Library/Fonts/Supplemental/Courier New.ttf",
	"/Library/Fonts/Courier New.ttf",
}

var (
	loadedFont *sfnt.Font
	loadOnce   sync.Once
	loadErr    error
)

// getFont loads and parses the system monospace font once.
func getFont() (*sfnt.Font, error) {
	loadOnce.Do(func() {
		var selectedPath string
		for _, path := range fontPaths {
			if _, err := os.Stat(path); err == nil {
				selectedPath = path
				break
			}
		}

		if selectedPath == "" {
			loadErr = errors.New("no system monospace font found")
			return
		}

		data, err := os.ReadFile(selectedPath)
		if err != nil {
			loadErr = err
			return
		}

		if strings.HasSuffix(strings.ToLower(selectedPath), ".ttc") {
			collection, err := opentype.ParseCollection(data)
			if err != nil {
				loadErr = err
				return
			}
			loadedFont, err = collection.Font(0)
			if err != nil {
				loadErr = err
				return
			}
		} else {
			loadedFont, err = opentype.Parse(data)
			if err != nil {
				loadErr = err
				return
			}
		}
	})

	return loadedFont, loadErr
}

// RenderTextToPNG renders input text into a lossless, non-anti-aliased monochrome PNG.
func RenderTextToPNG(text string) ([]byte, error) {
	fontObj, err := getFont()
	if err != nil {
		return nil, err
	}

	// Create a new font.Face locally to ensure thread-safety during concurrent rendering
	face, err := opentype.NewFace(fontObj, &opentype.FaceOptions{
		Size:    14,
		DPI:     72,
		Hinting: font.HintingNone,
	})
	if err != nil {
		return nil, err
	}
	defer face.Close()

	// 1. Process text lines and expand tabs
	rawLines := strings.Split(text, "\n")
	lines := make([]string, len(rawLines))
	maxLineLen := 0
	for idx, line := range rawLines {
		expanded := strings.ReplaceAll(line, "\t", "    ")
		lines[idx] = expanded
		if len(expanded) > maxLineLen {
			maxLineLen = len(expanded)
		}
	}

	// 2. Fetch metrics
	metrics := face.Metrics()
	ascent := metrics.Ascent.Ceil()
	fontLineHeight := metrics.Height.Ceil()
	
	// Default spacing factor: 1.25
	lineHeight := int(float64(fontLineHeight) * 1.25)
	
	advance, _ := face.GlyphAdvance('A')
	charWidth := advance.Ceil()
	if charWidth <= 0 {
		charWidth = 8 // Fallback standard Monaco width
	}

	margin := 20

	// 3. Compute image dimensions
	width := maxLineLen*charWidth + margin*2
	height := len(lines)*lineHeight + margin*2

	// Safe minimum bounds
	if width < 100 {
		width = 100
	}
	if height < 100 {
		height = 100
	}

	// 4. Create Grayscale (mode L) image canvas
	img := image.NewGray(image.Rect(0, 0, width, height))
	// Fill with white (255)
	draw.Draw(img, img.Bounds(), &image.Uniform{color.Gray{Y: 255}}, image.Point{}, draw.Src)

	// 5. Draw text line-by-line
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.Gray{Y: 0}), // Black text
		Face: face,
	}

	for idx, line := range lines {
		// Y coordinate is the baseline, which is top of line cell + ascent
		yBaseline := margin + idx*lineHeight + ascent
		d.Dot = fixed.Point26_6{
			X: fixed.I(margin),
			Y: fixed.I(yBaseline),
		}
		d.DrawString(line)
	}

	// 6. Thresholding: Remove anti-aliasing to enforce pixel-perfect sharp edges
	for i := 0; i < len(img.Pix); i++ {
		// Grayscale: 0 is black, 255 is white.
		// If a pixel has text coverage (is darker than 128), snap it strictly to black.
		if img.Pix[i] < 128 {
			img.Pix[i] = 0
		} else {
			img.Pix[i] = 255
		}
	}

	// 7. Encode to lossless PNG using BestSpeed configuration
	var buf bytes.Buffer
	encoder := png.Encoder{
		CompressionLevel: png.BestSpeed,
	}
	err = encoder.Encode(&buf, img)
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}
