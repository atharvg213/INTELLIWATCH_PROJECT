with open('frontend/js/dashboard.js', 'r', encoding='utf-8') as f:
    text = f.read()

target = 'function renderCanvas(data) {\n  const ctx = elements.canvas.getContext("2d");\n  const cw = elements.canvas.width;\n  const ch = elements.canvas.height;\n  const sw = data.frame_width || 1280;\n  const sh = data.frame_height || 720;\n  const sx = cw / sw;\n  const sy = ch / sy;'

replacement = 'function renderCanvas(data) {\n  const sw = data.frame_width || 1280;\n  const sh = data.frame_height || 720;\n  if (elements.canvas.width !== sw || elements.canvas.height !== sh) {\n    elements.canvas.width = sw;\n    elements.canvas.height = sh;\n  }\n  const ctx = elements.canvas.getContext("2d");\n  const cw = elements.canvas.width;\n  const ch = elements.canvas.height;\n  const sx = 1.0;\n  const sy = 1.0;'

assert target in text, "Target content not found in dashboard.js"
updated = text.replace(target, replacement, 1)

with open('frontend/js/dashboard.js', 'w', encoding='utf-8', newline='\n') as f:
    f.write(updated)

print("renderCanvas successfully updated!")
