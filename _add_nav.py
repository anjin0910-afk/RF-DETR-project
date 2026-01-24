from pathlib import Path

path = Path("new_website.html")
text = path.read_text(encoding="utf-8")
nav_block = "<ul class=\"nav-links text-sm\">"\ninsertion = "<li>\n<a href=\"#demo\">라이브 데모</a>\n</li>\n"
if nav_block not in text:
    raise SystemExit("nav list not found")
parts = text.split(nav_block, 1)
head = parts[0] + nav_block + "\n"
rest = parts[1]
if insertion in rest:
    raise SystemExit("nav item already present")
head += insertion
text = head + rest
path.write_text(text, encoding="utf-8")
