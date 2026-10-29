#!/bin/bash
# Usage: tools/md2pdf.sh path/to/report.md   -> writes report.pdf (and report.docx) next to it.
set -euo pipefail
md="$1"; dir="$(cd "$(dirname "$md")" && pwd)"; base="$(basename "$md" .md)"
css="$(cd "$(dirname "$0")" && pwd)/report.css"
cd "$dir"
pandoc "$base.md" -s --embed-resources --css "$css" -o "/tmp/$base.html" --metadata pagetitle="$base"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$dir/$base.pdf" "file:///tmp/$base.html" 2>/dev/null
pandoc "$base.md" -o "$base.docx"
rm -f "/tmp/$base.html"
echo "wrote $dir/$base.pdf and $base.docx"
