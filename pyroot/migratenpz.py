# migratenpz.py
import os
import shutil
import numpy as np

def migrate_history_npz(in_path: str, out_path: str):
    """迁移history.npz，缺失epochs自动补全"""
    raw = np.load(in_path, allow_pickle=True)
    data_dict = dict(raw)
    changed = False
    if "losses" in data_dict and "epochs" not in data_dict:
        n = len(data_dict["losses"])
        data_dict["epochs"] = np.arange(n)
        changed = True
        print(f"  ⚠ 补全epochs数组，长度={n}")
    np.savez(out_path, **data_dict)
    return changed

def process_one_round_folder(src_round_dir: str, dst_round_dir: str, inplace: bool = False):
    """
    处理单个Round文件夹
    src_round_dir: 源Round文件夹路径，例如 ./旧训练数据/Round1
    dst_round_dir: 输出Round文件夹路径
    inplace=True:原地修改源目录；False：拷贝到目标目录处理（推荐）
    """
    print(f"\n====处理 {os.path.basename(src_round_dir)} ====")
    if not inplace:
        # 完整复制整个Round文件夹到目标
        if os.path.exists(dst_round_dir):
            shutil.rmtree(dst_round_dir)
        shutil.copytree(src_round_dir, dst_round_dir)
        work_dir = dst_round_dir
    else:
        work_dir = src_round_dir

    # 1.处理history.npz
    hist_src = os.path.join(src_round_dir, "history.npz")
    hist_dst = os.path.join(work_dir, "history.npz")
    if os.path.isfile(hist_src):
        changed = migrate_history_npz(hist_src, hist_dst)
        if changed:
            print(f"✅ history.npz 已补全epochs字段")
        else:
            print(f"✅ history.npz字段完整无需修改")
    else:
        print(f"⚠ 本Round没有history.npz，跳过npz迁移")

    # 2. 生成DONE标记文件（新版本训练完成标志）
    done_file = os.path.join(work_dir, "DONE")
    if not os.path.exists(done_file):
        with open(done_file, "w", encoding="utf‑8") as f:
            f.write("finished\n")
        print(f"✅ 已生成DONE标记文件")
    else:
        print(f"ℹ DONE文件已存在，无需重复生成")

def batch_process_top_dir(src_top: str, dst_top: str, inplace=False):
    """
    src_top: 顶层目录，例如 "./旧训练数据"，下面包含 Round1 Round2 ...子文件夹
    dst_top: 输出顶层目录
    """
    if not os.path.isdir(src_top):
        print(f"源顶层目录不存在：{src_top}")
        return
    if not inplace:
        os.makedirs(dst_top, exist_ok=True)

    entries = os.listdir(src_top)
    round_folders = []
    for name in entries:
        full = os.path.join(src_top, name)
        if os.path.isdir(full) and name.startswith("Round"):
            round_folders.append(name)
    if len(round_folders) == 0:
        print("未找到任何Round*子文件夹")
        return
    round_folders.sort()
    print(f"检测到 {len(round_folders)} 个Round文件夹：{round_folders}")

    for r_name in round_folders:
        src_rd = os.path.join(src_top, r_name)
        if inplace:
            dst_rd = src_rd
        else:
            dst_rd = os.path.join(dst_top, r_name)
        process_one_round_folder(src_rd, dst_rd, inplace=inplace)

if __name__ == "__main__":
    # =====================配置区========================
    INPLACE_MODE = False
    # False=【推荐】复制一份做迁移，原始旧数据完全不动
    # True=直接原地修改旧目录，风险！务必先手动备份！

    SRC_TOP = r".\旧训练数据"
    DST_TOP = r".\migrated_训练数据"
    # ====================================================

    print("=== npz&目录迁移工具 ===")
    if INPLACE_MODE:
        ans = input("⚠开启原地模式，会直接修改源文件，确认？(y/n):").strip().lower()
        if ans != "y":
            print("已退出")
            exit(0)
    batch_process_top_dir(SRC_TOP, DST_TOP, inplace=INPLACE_MODE)
    print("\n🎉全部任务完成！")
    if not INPLACE_MODE:
        print(f"输出目录：{os.path.abspath(DST_TOP)}")
        print("提示：把migrated_训练数据作为程序的训练数据目录使用")
