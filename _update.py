from pathlib import Path

path = Path("new_website.html")
text = path.read_text(encoding="utf-8")
old_js = "        document.querySelectorAll('a[href^=\"#\"]').forEach(anchor => {\n            anchor.addEventListener('click', function (e) {\n                e.preventDefault();\n                const target = document.querySelector(this.getAttribute('href'));\n                if (target) {\n                    const headerHeight = 80;\n                    const targetPosition = target.offsetTop - headerHeight;\n                    window.scrollTo({\n                        top: targetPosition,\n                        behavior: 'smooth'\n                    });\n                }\n            });\n        });"
new_js = "        document.querySelectorAll('a[href^=\"#\"]').forEach(anchor => {\n            anchor.addEventListener('click', function (e) {\n                const target = document.querySelector(this.getAttribute('href'));\n                if (!target) {\n                    return;\n                }\n                e.preventDefault();\n                target.scrollIntoView({ behavior: 'smooth', block: 'start' });\n            });\n        });"
if old_js not in text:
    raise SystemExit('Original JS snippet not found')
text = text.replace(old_js, new_js)
old_section = "        .section {\n            padding: var(--space-16) 0;\n        }"
new_section = "        .section {\n            padding: var(--space-16) 0;\n            scroll-margin-top: 100px;\n        }"
if old_section in text:
    text = text.replace(old_section, new_section)
else:
    raise SystemExit('Section CSS not found')
old_metrics = "        .metrics-section {\n            background: var(--gradient-primary);"
new_metrics = "        .metrics-section {\n            scroll-margin-top: 100px;\n            background: var(--gradient-primary);"
if old_metrics in text:
    text = text.replace(old_metrics, new_metrics)
else:
    raise SystemExit('Metrics CSS not found')
path.write_text(text, encoding='utf-8')
