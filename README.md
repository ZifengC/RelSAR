# RelSAR

Minimal run instructions.

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Prepare data

Put the processed dataset under `data/`.

The code currently supports:
- `KuaiSAR`
- `Amazon`

## 3. Optional: build a static item graph

```bash
python3 utils/build_item_graph.py --data KuaiSAR --topk 16 --output data/item_graph_topk.pt
```

This prints a progress bar while building the graph.

## 4. Train

Without item graph:

```bash
python3 main.py --model UniSAR --data KuaiSAR
```

With item graph:

```bash
python3 main.py --model UniSAR --data KuaiSAR --item_graph_path data/item_graph_topk.pt
```

Use CPU only:

```bash
python3 main.py --model UniSAR --data KuaiSAR --gpu cpu
```

## 5. Test

```bash
python3 main.py --model UniSAR --data KuaiSAR --train 0 --test_path output/KuaiSAR/UniSAR/checkpoints/<run_name>/best.pt
```

## 6. Logs

Training logs are written to:

```text
output/<DATA>/<MODEL>/logs/<TIME>.log
```

The log includes:
- training loss
- validation/test results
- epoch-level diagnostics for uncertainty, counterfactual gate, and path strength
