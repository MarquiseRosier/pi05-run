# Pi0.5 local / Colab inference

Colab 适合冒烟。复杂实验（换 suite、换 task 0–9、整局换 prompt、做可解释统计）用这个目录。

场景由 `suite` + `task_id` 决定。`prompt` 才是喂给 VLA 的语言。成功判定仍是官方 LIBERO predicate，不会跟着自定义 prompt 改。

## 环境

Windows 和 Linux / Colab 都可以跑。脚本会按平台设 MuJoCo 后端：Windows 用 `wgl`，Linux/Colab 用 `egl`。需要 NVIDIA GPU、已装好的 `lerobot` + `libero`、Pi0.5 权重；probe/replace 还要 transcoder `.pt`。

PowerShell 示例：

```powershell
python run/pi0.5/infer.py run --suite libero_spatial --task-id 3 --prompt "pick up the cookie box and place it on the plate" --mode probe --transcoder-checkpoint C:\path\to\step_027233.pt
```

## 看官方任务

```bash
python run/pi0.5/infer.py list-tasks
python run/pi0.5/infer.py list-tasks --suite libero_spatial
```

`libero_spatial` 的 0–9 就是「同一个厨房布局里，黑碗在不同位置」那 10 个场景。

## 跑一整局，并随便换 prompt

官方语言（空 prompt）：

```bash
python run/pi0.5/infer.py run \
  --suite libero_spatial \
  --task-id 3 \
  --mode probe \
  --transcoder-checkpoint /path/to/step_027233.pt
```

自定义语言（场景仍是 task 3，VLA 读你的句子）：

```bash
python run/pi0.5/infer.py run \
  --suite libero_spatial \
  --task-id 3 \
  --prompt "pick up the cookie box and place it on the plate" \
  --mode probe \
  --transcoder-checkpoint /path/to/step_027233.pt
```

多个场景：

```bash
python run/pi0.5/infer.py run --suite libero_spatial --task-ids 0,1,3 --prompt ""
```

或改 `run/pi0.5/config.yaml` 后：

```bash
python run/pi0.5/infer.py run --config run/pi0.5/config.yaml
```

Colab 里同样命令即可，不必再走 `lerobot-eval`（那个入口不能改语言）。

## 输出

`outputs/pi05_infer/<run_id>/`

- `eval_info.json`：官方语言、实际 VLA prompt、success、步数
- `episodes/task_<id>_ep<n>/video.mp4`
- `transcoder_capture/events.jsonl`：每个 chunk 的 `z` 摘要（probe/replace）

`n_action_steps=10`，和 notebook / 官方 eval 一样。不要和 GR00T 的 8 混用。

## 按 prompt 过滤 L0% vs t

```bash
python run/pi0.5/analyze_l0.py --run outputs/pi05_infer/<run_id>
python run/pi0.5/analyze_l0.py --run outputs/pi05_infer/<run_id> --task-contains "cookie box"
```
