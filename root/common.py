# common.py 修改后完整关键函数
import os
import re
import json
import numpy as np
from typing import Tuple, Optional

TASK_DEFINITION = {
    "blom": {"label": "逻辑门", "folder_suffix": "训练数据"},
    "ecnm": {"label": "英文名字", "folder_suffix": "训练数据"},
    "pmm": {"label": "古诗", "folder_suffix": "训练数据"},
}
BASE_DATA_ROOT = "训练数据"

def get_task_dir(task_key: str) -> str:
    cfg = TASK_DEFINITION[task_key]
    return os.path.join(BASE_DATA_ROOT, f"{task_key}{cfg['folder_suffix']}")

def ensure_task_dir(task_key: str) -> str:
    td = get_task_dir(task_key)
    os.makedirs(td, exist_ok=True)
    return td

def find_unfinished(task_dir: str):
    """【已删除 (U) 逻辑】只看DONE文件，无DONE=未完成"""
    result = []
    for d in os.listdir(task_dir):
        full = os.path.join(task_dir, d)
        if not os.path.isdir(full):
            continue
        # 正则：只匹配 Round数字，彻底抛弃 (U)
        match = re.fullmatch(r"Round(\d+)", d)
        if not match:
            continue
        n = int(match.group(1))
        done_file = os.path.join(full, "DONE")
        # 没有DONE，判定未完成
        if not os.path.exists(done_file):
            result.append((n, full))
    result.sort()
    return result

def new_round_dir(task_dir: str) -> str:
    """永远生成 RoundX，不再有is_unfinished、不再拼接(U)"""
    existing = [d for d in os.listdir(task_dir) if re.fullmatch(r"Round\d+", d)]
    max_n = 0
    for d in existing:
        n = int(re.match(r"Round(\d+)", d).group(1))
        max_n = max(max_n, n)
    basename = f"Round{max_n +1}"
    new_path = os.path.join(task_dir, basename)
    os.makedirs(new_path, exist_ok=True)
    return new_path

def mark_round_finished(round_dir: str):
    """【重要：彻底移除文件夹重命名逻辑！只写DONE，不改文件夹名字！】"""
    done_flag = os.path.join(round_dir, "DONE")
    with open(done_flag, "w", encoding="utf-8") as f:
        f.write("finished\n")
    # ========= 删除原来os.rename去掉(U)的全部代码 =========

def load_history_npz(history_path: str) -> Tuple[int, list]:
    start_epoch = 0
    losses = []
    if os.path.exists(history_path):
        try:
            data = np.load(history_path, allow_pickle=True)
            if "losses" in data.files:
                losses = list(data["losses"])
                start_epoch = len(losses)
        except Exception:
            pass
    return start_epoch, losses

def save_config(round_dir: str, config: dict):
    cfg_path = os.path.join(round_dir, "config.json")
    if not os.path.exists(cfg_path):
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

def load_config(round_dir: str) -> Optional[dict]:
    cfg_path = os.path.join(round_dir, "config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
