# Ideogram 4 Prompting Guide

## What Ideogram 4 Is

Ideogram 4 is a **structured caption model** — not a free-text prompt model. Instead of one long string, you give it a JSON caption with spatial regions (bounding boxes), style controls, and per-element descriptions. The model composes the image region by region.

## JSON Caption Structure

The output of `Ideogram4PromptBuilderKJ` is a JSON object with this structure:

```json
{
    "high_level_description": "...",
    "style_description": {
        "aesthetics": "...",
        "lighting": "...",
        "photo": "...",
        "medium": "..."
    },
    "compositional_deconstruction": {
        "background": "...",
        "elements": [
            {
                "type": "obj",
                "bbox": [ymin, xmin, ymax, xmax],
                "desc": "...",
                "color_palette": ["#RRGGBB"]
            }
        ]
    }
}
```

### Top-Level Fields

| Field | Required | Description |
|-------|----------|-------------|
| `high_level_description` | No | One-line overview of the whole image. Omitted if blank. |
| `style_description` | No | Style controls. Omitted entirely if `style` is set to `none`. |
| `compositional_deconstruction` | **Yes** | The core of the prompt — background and elements. |

### Style Description

Set via the `style` widget: `none`, `photo`, or `art_style`.

**When `style = "photo"`**, the output keys are (in order):
- `aesthetics` — mood, quality, tone
- `lighting` — light quality and direction
- `photo` — camera/photography descriptors
- `medium` — artistic medium
- `color_palette` — optional, up to 16 hex colors

**When `style = "art_style"`**, the output keys are:
- `aesthetics`
- `lighting`
- `medium`
- `art_style` — the art style descriptor
- `color_palette` — optional

**When `style = "none"`**, no `style_description` is emitted at all.

### Compositional Deconstruction

| Field | Required | Description |
|-------|----------|-------------|
| `background` | **Yes** | Scene background description. Always emitted. |
| `elements` | No | Array of region objects. Can be empty. |

### Element Object

| Field | Required | Description |
|-------|----------|-------------|
| `type` | **Yes** | `"obj"` for visual elements, `"text"` for text regions |
| `bbox` | No | `[ymin, xmin, ymax, xmax]` on a **0–1000 grid**. Omitted for unplaced elements. |
| `desc` | No | Free-text description of what should appear in this region. |
| `text` | No | Exact text to render (only for `type: "text"`). |
| `color_palette` | No | Up to **5** hex color codes that guide the region's colors. |

## The 0–1000 BBox Grid

Bounding boxes use a normalized 0–1000 grid, **not** pixel coordinates.

Format: `[ymin, xmin, ymax, xmax]`

- `(0, 0)` is the **top-left** corner
- `(1000, 1000)` is the **bottom-right** corner
- `y` increases downward, `x` increases rightward

**Examples:**
- Top-left quadrant: `[0, 0, 500, 500]`
- Full width, bottom third: `[667, 0, 1000, 1000]`
- Center circle area: `[350, 350, 650, 650]`

The node converts internal normalized coordinates (0.0–1.0) to this grid:
```python
ymin = round(y * 1000)
xmin = round(x * 1000)
ymax = round((y + h) * 1000)
xmax = round((x + w) * 1000)
```

## Using the Ideogram4PromptBuilderKJ Node

### Inputs

| Input | Type | Purpose |
|-------|------|---------|
| `width` / `height` | INT | Canvas dimensions. Must be multiples of 16. |
| `high_level_description` | STRING | Optional overview line. |
| `background` | STRING | **Required.** Scene background. |
| `style` | COMBO | `none`, `photo`, or `art_style`. |
| `aesthetics` / `lighting` / `medium` | STRING | Style sub-fields. |
| `import_json` | STRING (wired) | Wire a full caption JSON from another node. |
| `import_mode` | COMBO | `when empty` (seed only, then editor wins) or `always` (wired JSON is authoritative). |
| `bboxes` | BoundingBox | Wire pixel-space boxes from SAM3 or detector nodes. |

### Toolbar Buttons

| Button | Action |
|--------|--------|
| **Copy** | Copies the current caption JSON to clipboard. |
| **Paste** | Reads clipboard (or prompts) for a caption JSON and loads it into the editor. |
| **Clear all** | Removes all regions. |
| **Grab BG** / **Clear BG** | Use last generated image as background reference. |
| **Live** | Use live sampling preview as background. |
| **compact** checkbox | Toggle compact vs pretty JSON output. Ideogram 4 expects compact. |

### Canvas Controls

| Action | Effect |
|--------|--------|
| Drag on canvas | Draw a new region |
| Ctrl+Drag | Force-draw over existing region |
| Click | Select a region |
| Alt+Click | Cycle overlapping regions |
| Double-click | Edit description inline |
| Right-click | Region list (select/delete/duplicate/reorder) |
| Del/Backspace | Remove selected region |
| Ctrl+C/V/D | Copy/paste/duplicate selected region |

## Importing JSON Captions

Two methods:

**1. Paste button (recommended):** Copy a caption JSON string to your clipboard, click the **Paste** button on the node toolbar. The JSON must contain `compositional_deconstruction`.

**2. Wired `import_json`:** Connect a `PrimitiveNode` or `ShowText|pysssss` output to the `import_json` input. Set `import_mode`:
- `when empty` — JSON seeds the editor only if it has no regions. Manual edits persist.
- `always` — Wired JSON is always authoritative; editor snaps back on every execution.

## Common Patterns

### Portrait with subject placement

```json
{
    "high_level_description": "A portrait of a woman sitting on a sofa",
    "style_description": {
        "aesthetics": "realistic, casual",
        "lighting": "overcast, calm",
        "photo": "amateur photography",
        "medium": "photography"
    },
    "compositional_deconstruction": {
        "background": "living room interior",
        "elements": [
            {
                "type": "obj",
                "bbox": [100, 50, 700, 600],
                "desc": "a woman sitting on a grey sofa",
                "color_palette": ["#B0BEC5"]
            },
            {
                "type": "obj",
                "bbox": [700, 100, 950, 500],
                "desc": "coffee table with house plants",
                "color_palette": ["#4CAF50"]
            }
        ]
    }
}
```

### Technical diagram

```json
{
    "high_level_description": "A technical flow diagram",
    "style_description": {
        "aesthetics": "modern minimal, clean, professional",
        "lighting": "even, flat, uniform",
        "photo": "technical illustration, rounded shapes, crisp typography",
        "medium": "digital illustration"
    },
    "compositional_deconstruction": {
        "background": "pure white background",
        "elements": [
            {
                "type": "obj",
                "bbox": [150, 0, 450, 300],
                "desc": "a rounded rectangle box in cool gray, labeled 'Input'",
                "color_palette": ["#B0BEC5"]
            },
            {
                "type": "text",
                "bbox": [340, 290, 380, 310],
                "text": "Arrow",
                "desc": "a dark arrow pointing right"
            }
        ]
    }
}
```

### Multi-subject scene with color control

```json
{
    "compositional_deconstruction": {
        "background": "dark city street at night, wet pavement reflections",
        "elements": [
            {
                "type": "obj",
                "bbox": [200, 100, 600, 400],
                "desc": "a neon sign glowing red",
                "color_palette": ["#E76F51", "#FF0000"]
            },
            {
                "type": "obj",
                "bbox": [200, 600, 600, 900],
                "desc": "a neon sign glowing blue",
                "color_palette": ["#56B4E9", "#0072B2"]
            }
        ]
    }
}
```

## Key Rules

1. **`compositional_deconstruction` is mandatory** — without it, the JSON is rejected.
2. **`bbox` is `[ymin, xmin, ymax, xmax]`** — not `[x, y, w, h]`. Y comes first.
3. **BBox values are 0–1000**, not 0.0–1.0 or pixel coordinates.
4. **Width/height must be multiples of 16.**
5. **Color palettes are capped:** 5 per element, 16 for style.
6. **`type: "text"` elements need both `text` and `desc`.** The `text` field is what gets rendered; `desc` guides the surrounding visual context.
7. **Unplaced elements** (no `bbox`) are omitted from output — they exist only as editor placeholders.
8. **Style key order matters.** For `photo`: aesthetics, lighting, photo, medium. For `art_style`: aesthetics, lighting, medium, art_style.
9. **Use compact output** for Ideogram 4 — the model was trained on compact JSON format.
10. **Token budget:** Rough estimate is chars/4. Grey <256, green is healthy, orange is nearing limit, red is >=2048 (model cap — will error).

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| "Not a valid Ideogram 4 caption JSON" | Missing `compositional_deconstruction` | Ensure the JSON has this top-level key |
| "Required input is missing: output_format" | Workflow saved before node added this widget | Set `widgets_values[13]` to `"compact"` in workflow JSON, or reload the node |
| Regions not appearing | BBox out of 0–1000 range | Clamp all bbox values to 0–1000 |
| Text not rendering | Missing `text` field on `type: "text"` element | Add the `text` field with the exact string to render |
| Wrong colors | No `color_palette` on element | Add hex colors to guide the region's palette |
| Output is pretty-printed but Ideogram errors | `output_format` is `"pretty"` | Check the `compact` checkbox in the toolbar |
