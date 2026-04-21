# Mamba3MIMO（Pure Triton 使用示例）

本仓库当前支持通过 `mimo_backend="triton"` 使用 Triton 路径运行 Mamba3。

> ⚠️ 现阶段 Triton MIMO 仅支持 `mimo_rank=1`。如果你需要多 rank（如 2/4/8），请先使用 `mimo_backend="tilelang"`。

---

## 1. 最小可运行示例（训练前向+反向）

```python
import torch
from mamba_ssm.modules.mamba3 import Mamba3

# 设备与 dtype
# - T4(SM75): 仅支持 fp16 / fp32
# - 其他 dtype 仅在对应硬件支持时可用

device = "cuda"
dtype = torch.float16

model = Mamba3(
    d_model=1024,
    d_state=128,
    expand=2,
    headdim=64,
    is_mimo=True,
    mimo_rank=1,             # Triton 路径当前仅支持 rank=1
    mimo_backend="triton",  # 关键：强制走 Triton
    chunk_size=16,
    angle_wrap_interval=256, # 长序列时将角度映射回 [-pi, pi]
    device=device,
    dtype=dtype,
).train()

x = torch.randn(2, 512, 1024, device=device, dtype=dtype, requires_grad=True)
y = model(x)
loss = y.float().square().mean()
loss.backward()

print("ok", y.shape)
```

---

## 2. 推理示例（仅前向）

```python
import torch
from mamba_ssm.modules.mamba3 import Mamba3

model = Mamba3(
    d_model=1024,
    d_state=128,
    headdim=64,
    is_mimo=True,
    mimo_rank=1,
    mimo_backend="triton",
    chunk_size=16,
    device="cuda",
    dtype=torch.float16,
).eval()

with torch.no_grad():
    x = torch.randn(1, 1024, 1024, device="cuda", dtype=torch.float16)
    y = model(x)
print(y.shape)
```

---

## 3. 后端切换方式

### 方式 A：构造参数指定（推荐）

```python
model = Mamba3(..., is_mimo=True, mimo_rank=1, mimo_backend="triton")
```

可选值：
- `"tilelang"`: 使用 Tilelang MIMO 内核
- `"triton"`: 使用 Triton MIMO 路径（当前 rank=1）
- `"auto"`: 当前默认映射到 `tilelang`

### 方式 B：环境变量指定

```bash
export MAMBA3_MIMO_BACKEND=triton
```

---

## 4. 约束与注意事项

1. **Triton MIMO 当前仅支持 `mimo_rank=1`**。  
2. **T4（SM75）只允许 `fp16/fp32`**；`bf16/float8` 等仅在支持的架构上允许。  
3. `chunk_size` 需要 `>= 8`。  
4. 默认启用角度稳定化参数 `angle_wrap_interval=256`（满足长度条件后会将角度回卷到 `[-pi, pi]`）。

---

## 5. 一致性验证（Phase 7）

可使用脚本比较 Tilelang 与 Triton（rank=1）的前向/反向一致性：

```bash
python mamba_ssm/ops/triton/mamba3/phase7_validate.py \
  --batch 2 --seqlen 512 --d-model 1024 --d-state 128 --headdim 64 \
  --chunk-size 16 --dtype fp16 --tol-out-max 1e-2 --tol-grad-max 1e-2
```

---

## 6. 基线性能采样（Phase 0）

```bash
python mamba_ssm/ops/triton/mamba3/phase0_baseline.py \
  --backend triton --mimo-ranks 1 --seqlens 1024,2048 --chunk-sizes 8,16 --dtype fp16
```

输出为 JSONL，包含前向/反向耗时与峰值显存。
