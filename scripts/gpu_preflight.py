import json
import torch
from pathlib import Path
from common import setup, load_data, make_model, autocast

device, precision = setup('cuda', 'bf16', 4)
load_data()
print(json.dumps({'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(), 'bf16': torch.cuda.is_bf16_supported()}), flush=True)
for config in ['student_matched', 'student_10']:
    model, _ = make_model('student', json.loads(Path('configs/' + config + '.json').read_text()), device)
    optimizer = torch.optim.AdamW(model.parameters())
    for _ in range(4):
        ids = torch.randint(0, 2048, (32, 257), device=device)
        with autocast(device, precision):
            loss = torch.nn.functional.cross_entropy(model(ids[:, :-1]).flatten(0, 1).float(), ids[:, 1:].flatten()) / 4
        loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    print(json.dumps({'config': config, 'loss': loss.item() * 4,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}), flush=True)
    del optimizer, model, loss, ids
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
