# Quantization and Model Verification

This folder contains the model export, verification, numerical reference, host C test, synthetic blob generator, and optional QAT tooling.

The training implementation stays at the repository root (`model.py`, `train.py`, etc.) and is intentionally not duplicated here.

## Layout

```text
quantization/
├── export_int8.py
├── gtcrn_ref.py
├── host_test.c
├── make_synthetic_blob.py
├── qat_finetune.py
├── verify_export.py
└── docs/
    └── GTCRN_tools_file_guide.md
```

## Export

From the repository root:

```bash
python quantization/export_int8.py     --checkpoint checkpoints_v1/best_model.pt     --repo .     --out embedded/main/gtcrn_int8.bin
```

## Verify

```bash
python quantization/verify_export.py     --checkpoint checkpoints_v1/best_model.pt     --blob embedded/main/gtcrn_int8.bin     --repo .     --eval_root eval_set     --dump_vectors     --vectors_out embedded/main/gtcrn_testvec.bin
```

## Host C test

```bash
cd quantization

gcc -O2 -std=c99     -I ../embedded/components/gtcrn/include     -o host_test     host_test.c     ../embedded/components/gtcrn/gtcrn_ops.c     ../embedded/components/gtcrn/gtcrn_net.c     -lm
```

Do not commit the generated `host_test` executable.
