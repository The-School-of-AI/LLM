# ✅ ZIP Package Checklist

## Current Status: READY TO ZIP 🎯

### ✅ Self-Contained Files (All in endGame/)

**Core Models:**
- ✅ `recurrence_model_1b.py` - 1B dense (0 experts, 8 layers)
- ✅ `recurrence_model_70b.py` - 70B sparse (2 experts, 8 layers, Mac-optimized)
- ✅ `reversible_ops_midpoint.py` - Dependency

**Training:**
- ✅ `train_recurrence_1b.py` - No external imports ✓
- ✅ `train_recurrence_70b.py` - No external imports ✓
- ✅ `data_utils.py` - Dataset loader

**Tokenizer:**
- ✅ `tokenizer.json` (7.5MB) - 131k vocab

**Documentation:**
- ✅ `README_DEPLOYMENT.md` - Main instructions for your team
- ✅ `FIX_APPLIED.md` - Dense model fixes
- ✅ `GRADIENT_FIX.md` - Gradient flow fix
- ✅ `SETUP_COMPLETE.md` - 1B setup
- ✅ `SETUP_70B.md` - 70B setup (2 experts)
- ✅ `README_TRAINING.md` - Training guide
- ✅ This file - Pre-ZIP checklist

---

## ⚠️ What Your Team Must Add

### 1. Dataset Directory
```
your_project/
├── endGame/           ← Your ZIP extracts here
└── synth_local_en/    ← Team adds this
```

OR modify training scripts:
```python
# Line 239 in train_recurrence_1b.py and train_recurrence_70b.py
local_path="../synth_local_en",  # Change to their dataset path
```

### 2. Python Environment
```bash
pip install torch transformers datasets psutil
```

---

## 🧪 Pre-ZIP Verification

Run these commands to verify everything works:

```bash
cd /Users/rohanshravan/TSAI/ValidationCheck/endGame

# Test 1: Verify imports
python3 -c "
from recurrence_model_1b import create_model_1b
from recurrence_model_70b import create_model_70b
from data_utils import SYNTHStream
print('✅ All imports work!')
"

# Test 2: Check 70B config is scaled
python3 -c "
from recurrence_model_70b import ModelConfig
c = ModelConfig()
assert c.num_layers == 8, 'Layers should be 8'
assert c.num_real_experts == 2, 'Experts should be 2'
print('✅ 70B config is Mac-optimized (2 experts, 8 layers)')
"

# Test 3: Check no external imports
grep -r "sys.path" *.py | grep -v "#" | wc -l
# Should output: 0 (no sys.path manipulations)
```

---

## 📦 Creating the ZIP

### Option 1: Include Everything
```bash
cd /Users/rohanshravan/TSAI/ValidationCheck
zip -r endGame_package.zip endGame/ -x "*.pyc" "**/__pycache__/*" "*.git*"
```

### Option 2: Essential Files Only (Smaller ZIP)
```bash
cd /Users/rohanshravan/TSAI/ValidationCheck/endGame
zip ../endGame_package.zip \
  recurrence_model_1b.py \
  recurrence_model_70b.py \
  reversible_ops_midpoint.py \
  train_recurrence_1b.py \
  train_recurrence_70b.py \
  data_utils.py \
  tokenizer.json \
  README_DEPLOYMENT.md \
  FIX_APPLIED.md \
  GRADIENT_FIX.md \
  SETUP_COMPLETE.md \
  SETUP_70B.md \
  README_TRAINING.md \
  ZIP_CHECKLIST.md
```

**Recommended:** Use Option 2 for a cleaner package (~10-15 MB)

---

## 📝 Important Notes for Your Team

### 1. 70B Model is Scaled Down
The `recurrence_model_70b.py` is **NOT production-ready**. It's scaled for Mac testing:

| Parameter | Production | This Package |
|-----------|------------|--------------|
| Layers | 20 | 8 |
| Experts | 254 | 2 |
| Memory | ~128 GB | ~3-4 GB |

**For production:** Restore original values (documented in SETUP_70B.md)

### 2. Dataset Not Included
The `synth_local_en/` dataset is **NOT** in the ZIP. Your team needs to either:
- Use their existing copy
- Download from Hugging Face
- Modify training scripts to use different dataset

### 3. First Run is Slow
- **Step 0**: 5-10 seconds (Metal shader compilation on Mac)
- **Steps 1+**: 100-200ms (1B) or 200-500ms (70B scaled)

This is normal for MPS on Mac. CUDA will be 10-50× faster.

### 4. Expected Behavior

**1B Model (Dense):**
```
aux: 0.0000  ← Always zero (no MoE routing)
```

**70B Model (Sparse):**
```
aux: 0.0256  ← NON-ZERO (MoE routing losses)
```

If 70B shows `aux: 0.0000`, something is wrong!

---

## 🐛 Common Issues

### "ImportError: No module named X"
Run from endGame directory:
```bash
cd endGame
python train_recurrence_1b.py
```

### "Dataset not found"
Either:
1. Place dataset at `../synth_local_en/`
2. Or create it with:
```bash
python download_mini_synth.py --output-dir ../synth_local_en --max-samples 5000
```
3. Or change `local_path` in training scripts

### "70B loads wrong config (254 experts)"
You edited the wrong file! The ZIP should have:
```bash
grep "num_real_experts" recurrence_model_70b.py
# Should show: num_real_experts = 2  # Reduced from 254
```

### "OOM during initialization"
The 70B model should use **2 experts, 8 layers** (not 254 experts, 20 layers).
Verify with:
```bash
python -c "from recurrence_model_70b import ModelConfig; print(ModelConfig().num_real_experts)"
# Should print: 2
```

---

## ✅ Final Checklist Before Sending

- [ ] Verified imports work (no external dependencies)
- [ ] Checked 70B config shows 2 experts, 8 layers
- [ ] Included tokenizer.json (7.5MB)
- [ ] Included all documentation (6 MD files)
- [ ] ZIP is < 20MB
- [ ] Tested that models load without errors
- [ ] Reviewed README_DEPLOYMENT.md for clarity

---

## 🚀 Ready to Share!

Your team will:
1. Extract the ZIP
2. Add their dataset (or modify dataset path)
3. Run `python train_recurrence_1b.py`
4. Run `python train_recurrence_70b.py`

Everything else is self-contained!

**Package created on:** Feb 12, 2026
**Models tested on:** Mac M1/M2 with 64GB RAM
**Training verified:** ✅ Both models load and train successfully
