"""CPU validation-only resource gate; no test scoring or parameter selection by test."""
import argparse
import json
from pathlib import Path
import time

import torch

from common import ROOT, load_data, make_model, setup
from evaluate import score
from training_utils import hardware_metadata, inference_asset_bytes, source_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'evidence/cpu_gate.json')
    args = parser.parse_args()
    device, precision = setup('cpu', 'fp32', 4)
    data = load_data()
    rows = []
    for implementation, name in [('model', 'baseline'), ('student', 'student_matched'),
                                  ('student', 'student_8'), ('student', 'student_10')]:
        torch.manual_seed(17)
        config = json.loads((ROOT / 'configs' / f'{name}.json').read_text())
        model, _ = make_model(implementation, config, device)
        model.eval()
        with torch.no_grad():
            model.predict_log_probs(torch.zeros(32, 256, dtype=torch.long))
        result = score(model, *data['validation'], device, precision)
        result.pop('window_nll_nats')
        sources = [ROOT/'model.py'] if implementation == 'model' else [ROOT/'student.py', ROOT/'student_model.py']
        row = dict(config=name, parameters=sum(p.numel() for p in model.parameters()),
                   inference_asset_bytes=inference_asset_bytes(model, sources), **result)
        row['baseline_time_ratio'] = result['seconds'] / (rows[0]['seconds'] if rows else result['seconds'])
        row['resource_gate_passed'] = (row['baseline_time_ratio'] <= 4 and row['inference_asset_bytes'] < 56 * 2**20
                                       and hardware_metadata()['peak_cpu_ram_gib'] < 4)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(threads=4, precision='fp32', split='validation',
                                          note='Random weights: resource/initial-score check, not trained quality.',
                                          models=rows, **hardware_metadata(), **source_metadata(ROOT)), indent=2) + '\n')


if __name__ == '__main__':
    main()
