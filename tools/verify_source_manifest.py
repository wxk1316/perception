#!/usr/bin/env python3
"""Verify the checked-in source/model manifest without ROS or extra libraries."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text())
errors = []
for name, expected in manifest['files'].items():
    path = root/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append(name)
if errors:
    raise SystemExit('Missing or changed files: ' + ', '.join(errors))
print('Verified', len(manifest['files']), 'source/model/document files.')
