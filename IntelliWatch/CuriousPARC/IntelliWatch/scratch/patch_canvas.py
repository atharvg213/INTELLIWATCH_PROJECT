with open('frontend/js/dashboard.js', 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

for i, l in enumerate(lines):
    if 'function renderCanvas(data)' in l:
        print('Found line at:', i)
        lines[i:i+8] = [
            'function renderCanvas(data) {',
            '  const sw = data.frame_width || 1280;',
            '  const sh = data.frame_height || 720;',
            '  if (elements.canvas.width !== sw || elements.canvas.height !== sh) {',
            '    elements.canvas.width = sw;',
            '    elements.canvas.height = sh;',
            '  }',
            '  const ctx = elements.canvas.getContext("2d");',
            '  const cw = elements.canvas.width;',
            '  const ch = elements.canvas.height;',
            '  const sx = 1.0;',
            '  const sy = 1.0;'
        ]
        break

with open('frontend/js/dashboard.js', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('Updated renderCanvas successfully!')
