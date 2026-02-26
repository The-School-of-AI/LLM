# 🚀 endGame - Self-Contained Training Package

## ✅ What's Included in This ZIP

All files needed to train the recurrence models are **self-contained** in this directory:

### Core Model Files
- ✅ `recurrence_model_1b.py` - 1B dense model (0 experts, 8 layers)
- ✅ `recurrence_model_70b.py` - 70B sparse model (2 experts, 8 layers - scaled for testing)
- ✅ `reversible_ops_midpoint.py` - Required dependency for memory-efficient backprop

### Training Scripts
- ✅ `train_recurrence_1b.py` - Training script for 1B model
- ✅ `train_recurrence_70b.py` - Training script for 70B model
- ✅ `data_utils.py` - Dataset loading utilities

### Tokenizer
- ✅ `tokenizer.json` (7.5MB) - 131k token vocabulary

### Documentation
- ✅ `FIX_APPLIED.md` - Dense model initialization fixes
- ✅ `GRADIENT_FIX.md` - Reversible integration gradient flow fix
- ✅ `SETUP_COMPLETE.md` - 1B model setup guide
- ✅ `SETUP_70B.md` - 70B model setup guide (2 experts, 8 layers)
- ✅ `README_TRAINING.md` - Training guide
- ✅ This file - Deployment instructions

---

## ⚠️ What Your Team Needs to Provide

### 1. Dataset: `synth_local_en`

The training scripts expect the dataset at:
```
../synth_local_en/
```

**If your team has this dataset:**
- Place it one directory above `endGame/`
- Structure should be:
  ```
  your_project/
  ├── endGame/           ← This ZIP
  │   ├── train_recurrence_1b.py
  │   └── ...
  └── synth_local_en/    ← Dataset here
      ├── data files...
  ```

**If your team does NOT have this dataset:**
- They can download it from Hugging Face: `PleIAs/SYNTH`
- Or modify the training scripts to use a different dataset
- Change line 239 in both training scripts:
  ```python
  local_path="../synth_local_en",  # Change this path
  ```

### 2. Python Environment

Install required packages:
```bash
pip install torch transformers datasets psutil
```

**Recommended versions:**
- torch >= 2.0.0
- transformers >= 4.30.0
- datasets >= 2.12.0

---

## 🎯 Quick Start (After Extracting ZIP)

### Step 1: Verify Setup
```bash
cd endGame
python -c "from recurrence_model_1b import create_model_1b; print('✓ Imports work!')"
```

### Step 2: Train 1B Model (Dense)
```bash
python train_recurrence_1b.py
```

**Expected output:**
```
🤖 MODEL-1B (DENSE) INITIALIZED:
   Total Layers: 8 (6 DeltaNet + 2 GSA)
   Experts: 0 (dense model - no routing)
   Active Parameters: ~1.513B (100% active)

step   0 | loss_ntp: 12.60 | loss_mtp: 12.65 | aux: 0.0000 | dt: 5000ms
step  10 | loss_ntp:  9.23 | loss_mtp:  9.19 | aux: 0.0000 | dt:  150ms
```

### Step 3: Train 70B Model (Sparse MoE, scaled to 2 experts)
```bash
python train_recurrence_70b.py
```

**Expected output:**
```
🤖 MODEL-70B INITIALIZED:
   Total Layers: 8 (6 DeltaNet + 2 GSA)
   Experts: 2 real + 2 null = 4 slots
   Active Parameters: ~1.5B

step   0 | loss_ntp: 12.60 | loss_mtp: 12.65 | aux: 0.0256 | dt: 8000ms
                                                      ^^^^^ NON-ZERO for MoE
```

---

## 🔧 Configuration Changes from Production

### 70B Model Scaled Down for Testing

The `recurrence_model_70b.py` has been **modified from production** for Mac/local testing:

| Parameter | Production | This ZIP (Testing) |
|-----------|------------|-------------------|
| **num_layers** | 20 | 8 |
| **num_real_experts** | 254 | 2 |
| **num_null_experts** | 254 | 2 |
| **top_k** | 10 | 2 |
| **Memory** | ~128 GB | ~3-4 GB |

**⚠️ For production GPU training:**
Restore these values in `recurrence_model_70b.py` lines 377-404:
```python
num_layers = 20  # Was 8 for testing
num_deltanet_layers = 15
num_gsa_layers = 5
num_real_experts = 254  # Was 2 for testing
num_null_experts = 254
top_k = 10
```

---

## 📊 Model Comparison

| Feature | 1B Model | 70B Model (Testing) |
|---------|----------|---------------------|
| **Architecture** | Dense FFN | Sparse MoE |
| **Experts** | 0 | 2 |
| **Layers** | 8 | 8 |
| **Params** | 1.513B | ~1.5B (scaled) |
| **Active %** | 100% | ~50% (routing) |
| **aux_loss** | 0.0000 | 0.01-0.10 |
| **Use case** | Baseline testing | MoE routing verification |

---

## 🐛 Troubleshooting

### "No module named 'recurrence_model_1b'"
**Solution:** Make sure you're running from the `endGame/` directory:
```bash
cd endGame
python train_recurrence_1b.py
```

### "Dataset not found: ../synth_local_en"
**Solution:**
1. Check if dataset exists: `ls ../synth_local_en`
2. If not, either:
   - Download from Hugging Face: `PleIAs/SYNTH`
   - Or change `local_path` in training scripts

### "OOM (Out of Memory)"
**For 1B model:**
- Reduce `batch_size` in script (line 217): `batch_size=1`
- Reduce `seq_len` (line 216): `seq_len=32`

**For 70B model:**
- Already scaled to 2 experts + 8 layers
- If still OOM, use CPU: `device = torch.device("cpu")`

### "aux_loss is 0.0000 for 70B model"
**Problem:** MoE routing not working
**Check:**
1. Verify config: `python -c "from recurrence_model_70b import ModelConfig; print(ModelConfig().num_real_experts)"`
2. Should print `2` (not `0` or `254`)

### "MTP loss lower than NTP loss"
**This is expected!** MTP uses teacher forcing (gets true t+1 embedding to predict t+2), making it easier during training. At inference, only NTP is used.

---

## 🔑 Key Features Implemented

### ✅ Fixes Applied (All 43 fixes from CHANGELOG)
1. **Fix #42**: RoPE dtype casting (prevents float32/bf16 mismatches)
2. **Fix #43**: RMSNorm fp32 statistics (stability at 256k context)
3. **Dense model fixes**: Gradient flow through reversible integration
4. **MoE routing losses**: Load balancing, Z-loss, null-rate regularization

### ✅ Architecture
- **Hybrid**: 75% DeltaNet (O(N) linear) + 25% GSA (adaptive sparse)
- **Memory recurrence**: Infinite context via stream 3
- **Multi-Token Prediction**: NTP (t+1) + MTP (t+2) with teacher forcing
- **Kronecker embeddings**: Byte-level encoding for 131k vocab
- **YARN RoPE**: Scales to 256k context (current testing at 64 seq_len)

---

## 📦 File Manifest

```
endGame/
├── README_DEPLOYMENT.md           ← You are here
├── FIX_APPLIED.md                 ← Dense model initialization fixes
├── GRADIENT_FIX.md                ← Reversible integration fix
├── SETUP_COMPLETE.md              ← 1B model documentation
├── SETUP_70B.md                   ← 70B model documentation (2 experts)
├── README_TRAINING.md             ← Training guide
│
├── recurrence_model_1b.py         ← 1B dense model (0 experts, 8 layers)
├── recurrence_model_70b.py        ← 70B sparse model (2 experts, 8 layers)
├── reversible_ops_midpoint.py     ← Required dependency
│
├── train_recurrence_1b.py         ← 1B training script
├── train_recurrence_70b.py        ← 70B training script
├── data_utils.py                  ← Dataset utilities
│
├── tokenizer.json                 ← 131k token vocabulary (7.5MB)
│
└── (other legacy files)           ← Can be ignored for recurrence training
    ├── main.py                    ← Old training script
    ├── training.py                ← Old training utilities
    ├── model_gated_multitoken.py  ← Old model architecture
    └── ...
```

---

## 🚀 Ready to Train!

The package is **fully self-contained** except for:
1. Dataset (`synth_local_en/`) - your team needs to provide
2. Python packages - install via pip

All model code, fixes, and documentation are included. No external dependencies on "Recurrance Code" or other directories.

**For questions or issues:**
- Check the troubleshooting section above
- Review the setup guides (SETUP_COMPLETE.md, SETUP_70B.md)
- Verify imports work: `python -c "from recurrence_model_1b import create_model_1b; print('OK')"`

Good luck with training! 🎯
