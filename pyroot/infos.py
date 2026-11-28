# infos.py
import json
import re
import os
from common import BASE_DATA_ROOT, TASK_DEFINITION

def rebuild_info(base=BASE_DATA_ROOT):
    tasks = [ (k,v["label"]) for k,v in TASK_DEFINITION.items() ]
    lines = []
    for task, label in tasks:
        task_dir = os.path.join(base, f"{task}{TASK_DEFINITION[task]['folder_suffix']}")
        lines.append(f"=== {task}训练数据（{label}）===")
        if not os.path.isdir(task_dir):
            lines.append("（暂无）")
            lines.append("")
            continue
        rounds = sorted(
            [d for d in os.listdir(task_dir)
            if os.path.isdir(os.path.join(task_dir, d)) and re.fullmatch(r"Round\d+", d)],
            key=lambda x: int(re.match(r"Round(\d+)", x).group(1))
        )
        if not rounds:
            lines.append("（暂无）")
            lines.append("")
            continue
        for r in rounds:
            full = os.path.join(task_dir, r)
            status = "已完成" if os.path.exists(os.path.join(full, "DONE")) else "未完成"
            detail = ""
            cfg_path = os.path.join(full, "config.json")
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, encoding="utf-8") as f:
                        cfg = json.load(f)
                    keys = ["gate", "hidden_size", "lr", "epochs", "seed", "data_lines"]
                    parts = [f"{k}={cfg[k]}" for k in keys if k in cfg]
                    detail = "  " + "  ".join(parts)
                except Exception:
                    detail = "  (config 读取失败)"
            lines.append(f"{r:<12} {status}{detail}")
        lines.append("")
    info_path = os.path.join(base, "info.txt")
    os.makedirs(base, exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return info_path
