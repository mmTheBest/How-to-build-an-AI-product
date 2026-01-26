#!/usr/bin/env bash
set -euo pipefail

# Base book files
[ -f book/book.toml ]
[ -f book/src/README.md ]
[ -f book/src/SUMMARY.md ]

# Chapter files must exist
files=(
  book/src/part-1-choose-model/01-product-constraints.md
  book/src/part-1-choose-model/02-model-landscape.md
  book/src/part-1-choose-model/03-selection-framework.md
)

for f in "${files[@]}"; do
  [ -f "$f" ]
  # Must have H1 title and at least 4 H2 headings
  grep -q "^# " "$f"
  test "$(grep -c "^## " "$f")" -ge 4
  # Must include Opening, Example, Checklist, Takeaway sections
  grep -q "^## Opening" "$f"
  grep -q "^## Example" "$f"
  grep -q "^## Checklist" "$f"
  grep -q "^## Takeaway" "$f"

done
