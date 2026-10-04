import re
with open('frontend/index.html', encoding='utf-8') as f:
    html = f.read()

tabs = re.findall(r'data-tab=["\'](.*?)["\']', html)
print('Tabs found in index.html:', set(tabs))

cards = re.findall(r'id=["\'](stat\w+|metric\w+|badge\w+|tab\w+)["\']', html)
print('Stat/metric IDs in index.html:', cards[:15])
