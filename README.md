# nps-sro-ml
Machine Learning for NPS Streaming Readout

[Full documentation](https://jeffersonlab.github.io/nps-sro-ml/)

## Baseline Object Condensation evaluation

Evaluate an existing `ObjectCondensationAttn` checkpoint on held-out `.pt` graphs
with the repository's default Object Condensation decoder configuration:

```bash
python pytorch_src/evaluate_baseline.py \
  --checkpoint path/to/model_best.pth \
  --data-dir path/to/test_data \
  --batch-size 8 \
  --device cuda \
  --output-dir results/baseline_oc_attention
```

The command writes aggregate metrics as JSON, event and cluster details as CSV,
a text summary, and diagnostic plots. Decoder and matching thresholds are exposed
for controlled future studies; the baseline defaults are `--beta-thres 0.4`,
`--dist-thres 0.8`, and `--match-iou-threshold 0.5`. Run
`python pytorch_src/evaluate_baseline.py --help` for all options.
