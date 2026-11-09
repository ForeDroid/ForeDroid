#!/usr/bin/env python3
"""Print saved yearly detection results in a read-only terminal table."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata

from output_layout import TEST

ANSI = re.compile(r'\x1b\[[0-9;]*m')
COLUMNS = [('Year', 24), ('F1 score', 28)]


def width(text):
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1
               for c in ANSI.sub('', str(text)))


def pad(text, size, right=False):
    spaces = ' ' * max(0, size - width(text))
    return spaces + text if right else text + spaces


def number(value, percent=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return '—'
    return f'{value:.2%}' if percent else f'{value:.4f}'


def aggregate(rows):
    """Use pooled confusion counts, never average yearly F1 values."""
    if not rows:
        return {}
    result = {key: sum(row.get(key, 0) for row in rows)
              for key in ('total_apks', 'scorable_apks', 'TN', 'FP', 'FN', 'TP')}
    tn, fp, fn, tp = (result[key] for key in ('TN', 'FP', 'FN', 'TP'))
    total, scored = result['total_apks'], result['scorable_apks']
    result['coverage'] = scored / total if total else None
    result['f1'] = (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0) if scored else None
    result['malware_only_f1'] = 2 * tp / (2 * tp + fn) if tp + fn else None
    result['benign_fpr'] = fp / (tn + fp) if tn + fp else None
    return result


def render(data, years, color=False, gpmalware=None, fdroid=False, include_gpmalware=None):
    def ink(text, code):
        return f'\033[{code}m{text}\033[0m' if color else text
    edge = lambda text: ink(text, '38;5;240')
    cyan = lambda text: ink(text, '1;38;5;81')
    dim = lambda text: ink(text, '38;5;245')
    cells_width = sum(size for _, size in COLUMNS) + 3 * (len(COLUMNS) - 1)
    inner = cells_width + 2
    lines = []

    def rule(left, middle, right):
        lines.append(edge(left + middle * inner + right))

    def line(text):
        lines.append(edge('│') + ' ' + pad(text, cells_width) + ' ' + edge('│'))

    def row(values, header=False, total=False):
        values = [pad(str(value), size, right=i > 0 and not header)
                  for i, ((_, size), value) in enumerate(zip(COLUMNS, values))]
        if header:
            values = [dim(value) for value in values]
        else:
            values[0] = cyan(values[0]) if total else values[0]
            values[1] = cyan(values[1])
        line(edge(' │ ').join(values))

    rule('╭', '─', '╮')
    line(cyan('FOREDROID  ·  Yearly detection results'))
    training = 'F-Droid benign' if fdroid else '2014–2017 benign'
    line(dim(f'Training: {training}'))
    line(dim('Testing: ' + ' / '.join(map(str, years))))
    rule('├', '─', '┤')
    row([title for title, _ in COLUMNS], header=True)
    rule('├', '─', '┤')
    available = []
    for year in years:
        metrics = data['years'].get(str(year))
        if not isinstance(metrics, dict) or 'total_apks' not in metrics:
            row([str(year), 'No results yet'])
            continue
        available.append(metrics)
        row([str(year), number(metrics.get('malware_only_f1'))])
    if available:
        metrics = aggregate(available)
        rule('├', '─', '┤')
        row(['Total' if len(available) == len(years) else 'Available total',
             number(metrics['malware_only_f1'])], total=True)
    rule('╰', '─', '╯')
    lines += ['', '  F1 score uses malware samples only (precision = 1); benign false positives are excluded.',
              '  Only scorable APKs are included. Totals pool confusion counts across selected years.']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Show saved yearly F1 scores.')
    parser.add_argument('--output', type=Path, default=TEST, help='Test output root directory')
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2018, 2023)),
                        help='Years to display, e.g. --years 2018 2020')
    parser.add_argument('--fdroid', action='store_true', help='Show evaluation results for the F-Droid reference library')
    parser.add_argument('--no-color', action='store_true', help='Disable terminal colors')
    parser.add_argument('--gpmalware-output', type=Path, help='GPMalware output root; defaults to RQ2/GPMalware beside the yearly output directory')
    args = parser.parse_args()
    path = args.output.expanduser() / 'anomaly_detection_pipeline/results/metrics.json'
    if args.fdroid:
        path = args.output.expanduser().parent / 'F-Droid/testing/2018-2022/results/metrics.json'
    gp_path = (args.gpmalware_output.expanduser() if args.gpmalware_output else args.output.expanduser().parent/'RQ2/GPMalware')/'detection/metrics.json'
    gp_data, gp_error = None, None
    try:
        if not args.fdroid:
            gp_data = json.loads(gp_path.read_text(encoding='utf-8'))
            if not isinstance(gp_data, dict) or not isinstance(gp_data.get('scorable_only'), dict):
                raise ValueError('Missing GPMalware detection metrics: scorable_only')
    except (OSError, ValueError, TypeError) as error:
        gp_data, gp_error = None, str(error)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict) or not isinstance(data.get('years'), dict):
            raise ValueError('Missing yearly metrics: years; generate detection results first')
        text = render(data, sorted(set(args.years)),
                      color=sys.stdout.isatty() and not args.no_color and 'NO_COLOR' not in os.environ,
                      gpmalware=gp_data, fdroid=args.fdroid)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f'Cannot read detection results: {error}\nPath: {path}\n')
    print(text)
    print(f'\n  Data source: {path}\n')
    if not args.fdroid:
        print(f'  GPMalware: {gp_path}')
    if gp_error:
        print(f'  Could not read GPMalware metrics: {gp_error}')


if __name__ == '__main__':
    main()
