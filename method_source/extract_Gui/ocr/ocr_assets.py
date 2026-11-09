#!/usr/bin/env python3
"""Resolve each Backstage widget's drawables and recognize embedded text.

Only raster resources referenced by the widget (including XML drawable aliases,
selectors and layers) are read. No unrelated image text is added to a widget.
"""
import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps

RASTER = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}


class Resources:
    def __init__(self, app):
        self.app = Path(app).resolve()
        self.index = {}
        self.aliases = {}
        res = self.app / 'res'
        for folder in sorted(res.glob('*')):
            kind = folder.name.split('-')[0]
            if kind in ('drawable', 'mipmap'):
                for p in sorted(folder.iterdir()):
                    name = p.name.split('.')[0]
                    self.index.setdefault((kind, name), []).append(p)
            elif kind == 'values':
                for p in sorted(folder.glob('*.xml')):
                    try:
                        root = ET.parse(p).getroot()
                    except ET.ParseError:
                        continue
                    for e in root:
                        if e.tag == 'item' and e.get('type') in ('drawable', 'mipmap'):
                            self.aliases.setdefault((e.get('type'), e.get('name')), []).append(e.text or '')

    def resolve(self, ref, seen=None):
        # Backstage stores bare filenames as well as Android resource names.
        ref = ref.strip()
        if not ref or ref in ('NO_ICON', 'null') or ref.startswith('@android:'):
            return []
        match = re.fullmatch(r'@?(?:(drawable|mipmap)/)?([\w.-]+)', ref)
        if not match:
            return []
        kind, name = match.groups()
        name = name.split('.')[0]
        keys = [(kind, name)] if kind else [('drawable', name), ('mipmap', name)]
        seen = set() if seen is None else set(seen)
        found = set()
        for key in keys:
            if key in seen:
                continue
            branch = seen | {key}
            for alias in self.aliases.get(key, []):
                found.update(self.resolve(alias, branch))
            for p in self.index.get(key, []):
                if p.suffix.lower() in RASTER:
                    found.add(p)
                elif p.suffix == '.xml':
                    try:
                        root = ET.parse(p).getroot()
                    except ET.ParseError:
                        continue
                    for e in root.iter():
                        for value in e.attrib.values():
                            if value.startswith(('@drawable/', '@mipmap/')):
                                found.update(self.resolve(value, branch))
        return sorted(found)


class Recognizer:
    def __init__(self, executable='tesseract', language='eng', confidence=60, timeout=20):
        self.executable = executable
        self.language = language
        self.confidence = confidence
        self.timeout = timeout
        self.cache = {}
        if not shutil.which(executable):
            raise RuntimeError('Tesseract is missing; install tesseract-ocr and its language data')
        result = subprocess.run([executable, '--list-langs'], capture_output=True, text=True, check=True)
        missing = set(language.split('+')) - set(result.stdout.splitlines())
        if missing:
            raise RuntimeError('Missing Tesseract language data: ' + ', '.join(sorted(missing)))

    def read(self, path):
        if path in self.cache:
            return self.cache[path]
        result = {'image': str(path), 'text': '', 'status': 'no_text'}
        try:
            with Image.open(path) as image:
                image = image.convert('RGBA')
                # Nine-patch border pixels are control markers, not visible UI.
                if path.name.endswith('.9.png') and min(image.size) > 2:
                    image = image.crop((1, 1, image.width - 1, image.height - 1))
                white = Image.new('RGBA', image.size, 'white')
                white.alpha_composite(image)
                image = ImageOps.grayscale(white.convert('RGB'))
                scale = min(3.0, 2400 / max(image.size))
                image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
                image = ImageOps.expand(image, border=12, fill='white')
                with tempfile.TemporaryDirectory(prefix='gui-ocr-') as tmp:
                    source = Path(tmp) / 'input.png'
                    image.save(source)
                    run = subprocess.run(
                        [self.executable, str(source), 'stdout', '-l', self.language,
                         '--psm', '11', 'tsv'], capture_output=True, text=True,
                        timeout=self.timeout, check=True)
            words = []
            scores = []
            for row in csv.DictReader(io.StringIO(run.stdout), delimiter='\t'):
                text = (row.get('text') or '').strip()
                score = float(row.get('conf', '-1'))
                if text and any(char.isalnum() for char in text) and score >= self.confidence:
                    words.append(text)
                    scores.append(score)
            result['text'] = ' '.join(words)
            result['confidence'] = sum(scores) / len(scores) if scores else None
            result['status'] = 'ok' if words else 'no_text'
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            result.update(status='error', error=str(error))
        self.cache[path] = result
        return result


def enrich(app_dir, requests, output, recognizer):
    resources = Resources(app_dir)
    with open(requests, newline='', encoding='utf-8') as source, \
         open(output, 'w', newline='', encoding='utf-8') as target:
        writer = csv.writer(target, delimiter='\t')
        writer.writerow(['uid', 'ocr_text'])
        for row in csv.DictReader(source, delimiter='\t'):
            images = set()
            refs = json.loads(row['drawables'])
            for ref in refs:
                images.update(resources.resolve(ref))
            results = [recognizer.read(p) for p in sorted(images)]
            texts = list(dict.fromkeys(r['text'] for r in results if r['text']))
            text = ' '.join(texts)
            writer.writerow([row['uid'], text])
    errors = [r for r in recognizer.cache.values() if r['status'] == 'error']
    if errors:
        raise RuntimeError(f'{len(errors)} image(s) failed OCR: ' + '; '.join(r['image'] + ': ' + r.get('error', '') for r in errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--app-dir', required=True)
    parser.add_argument('--requests', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tesseract', default='tesseract')
    parser.add_argument('--language', default='eng')
    parser.add_argument('--confidence', type=float, default=60)
    parser.add_argument('--timeout', type=float, default=20)
    args = parser.parse_args()
    enrich(args.app_dir, args.requests, args.output,
           Recognizer(args.tesseract, args.language, args.confidence, args.timeout))


if __name__ == '__main__':
    main()
