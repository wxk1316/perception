#!/usr/bin/env python3
"""Fetch the fixed, public CenterPoint ONNX bundle without installing software."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

REVISION = '494c8171def40bd36cc2feb323e0a5acbfab132b'
ROOT_URL = 'https://huggingface.co/AutowareFoundation/lidar_centerpoint/resolve/' + REVISION
FILES = ('pts_voxel_encoder.onnx', 'pts_backbone_neck_head.onnx',
         'ml_package.param.yaml', 'detection_class_remapper.param.yaml')


def prepare(destination, variant='base'):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in (*FILES, 'README.md','LICENSE-2.0.txt'):
        url = ('https://www.apache.org/licenses/LICENSE-2.0.txt' if name=='LICENSE-2.0.txt' else
               ROOT_URL + ('/' if name == 'README.md' else '/' + variant + '/') + name)
        target = destination / name
        with tempfile.NamedTemporaryFile(dir=destination, delete=False) as output:
            temporary = Path(output.name)
            try:
                digest = hashlib.sha256()
                with urllib.request.urlopen(url, timeout=45) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
                output.close()
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        entries.append({'file': name, 'url': url, 'bytes': target.stat().st_size,
                        'sha256': digest.hexdigest()})
        print('Prepared', target, flush=True)
    (destination / 'bundle.lock.json').write_text(json.dumps({
        'repository': 'AutowareFoundation/lidar_centerpoint',
        'revision': REVISION, 'variant': variant, 'files': entries,
        'license': 'Apache-2.0; retain training-data terms in README.md',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--variant', choices=('base', 'tiny'), default='base')
    args = parser.parse_args()
    prepare(args.destination, args.variant)
