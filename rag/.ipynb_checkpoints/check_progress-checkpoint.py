# check_progress.py
"""
学习进度一键验证脚本
运行方式：cd rag && python check_progress.py
"""

import os
import sys
import json
import importlib
import traceback

RAG_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 工具函数
# ============================================================
def ok(msg):    print(f"  ✅ {msg}")
def fail(msg):  print(f"  ❌ {msg}")
def warn(msg):  print(f"  ⚠️  {msg}")
def section(title): print(f"\n{'='*55}\n{title}\n{'='*55}")

def check_file(rel_path, desc=""):
    full = os.path.join(RAG_ROOT, rel_path)
    exists = os.path.isfile(full)
    size = os.path.getsize(full) if exists else 0
    label = desc or rel_path
    if exists:
        ok(f"{label}  ({size/1024:.1f} KB)")
    else:
        fail(f"{label}  [文件不存在]")
    return exists

def check_dir(rel_path, desc=""):
    full = os.path.join(RAG_ROOT, rel_path)
    exists = os.path.isdir(full)
    label = desc or rel_path
    if exists:
        files = os.listdir(full)
        ok(f"{label}  ({len(files)} 个文件)")
    else:
        fail(f"{label}  [目录不存在]")
    return exists

def check_json(rel_path, desc="", min_count=None):
    full = os.path.join(RAG_ROOT, rel_path)
    label = desc or rel_path
    if not os.path.isfile(full):
        fail(f"{label}  [文件不存在]")
        return False
    try:
        with open(full) as f:
            data = json.load(f)
        count = len(data) if isinstance(data, list) else "dict"
        if min_count and isinstance(data, list) and len(data) < min_count:
            warn(f"{label}  (仅 {count} 条，期望 ≥ {min_count})")
            return False
        ok(f"{label}  ({count} 条)")
        return True
    except Exception as e:
        fail(f"{label}  [JSON解析失败: {e}]")
        return False

def check_import(module_name, install_hint=""):
    try:
        importlib.import_module(module_name)
        ok(f"import {module_name}")
        return True
    except ImportError as e:
        hint = f"  → pip install {install_hint}" if install_hint else ""
        fail(f"import {module_name}  [{e}]{hint}")
        return False

def try_import_local(rel_path, module_desc):
    """尝试导入本地模块，捕获语法错误和运行时错误"""
    full = os.path.join(RAG_ROOT, rel_path)
    if not os.path.isfile(full):
        fail(f"{module_desc}  [文件不存在，无法导入]")
        return False
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_check_mod", full)
        mod  = importlib.util.module_from_spec(spec)
        # 不执行 __main__，只做语法/导入检查
        spec.loader.exec_module(mod)
        ok(f"{module_desc}  [语法正常，可导入]")
        return True
    except SystemExit:
        ok(f"{module_desc}  [语法正常，含main入口]")
        return True
    except Exception as e:
        tb = traceback.format_exc().strip().split('\n')[-1]
        fail(f"{module_desc}  [导入错误: {tb}]")
        return False

# ============================================================
# 检查：原有基础文件
# ============================================================
section("【基础文件】原有代码（改动前基线）")
check_file("retriever.py",     "retriever.py（原有检索器）")
check_file("rag_pipeline.py",  "rag_pipeline.py（原有Pipeline）")
check_dir("law_corpus",        "law_corpus/（语料库）")
check_dir("vector_store",      "vector_store/（向量存储）")
check_file("vector_store/faiss.index",     "  └─ faiss.index")
check_file("vector_store/chunks_meta.json","  └─ chunks_meta.json")

# ============================================================
# 检查：依赖包安装
# ============================================================
section("【依赖检查】pip 包安装状态")
check_import("langchain_text_splitters",  "langchain-text-splitters==0.2.4")
check_import("chromadb",                  "chromadb==0.4.24")
check_import("rank_bm25",                 "rank-bm25==0.2.2")
check_import("sentence_transformers",     "sentence-transformers")
check_import("faiss",                     "faiss-cpu")
check_import("jieba",                     "jieba")
check_import("openai",                    "openai")

# ============================================================
# 检查：Phase A 文件结构
# ============================================================
section("【Phase A】切分策略实验")
a1 = check_file("phase_a/chunking_strategies.py", "chunking_strategies.py")
a2 = check_file("phase_a/evaluate_chunking.py",   "evaluate_chunking.py")
a3 = check_json("phase_a/chunk_results/chunks_A.json", "切分结果 chunks_A.json（策略A）", min_count=10)
a4 = check_json("phase_a/chunk_results/chunks_B.json", "切分结果 chunks_B.json（策略B）", min_count=10)
a5 = check_json("phase_a/chunk_results/chunks_C.json", "切分结果 chunks_C.json（策略C）", min_count=10)
a6 = check_file("phase_a/results/chunking_eval.json",  "评测结果 chunking_eval.json")

print()
if a1: try_import_local("phase_a/chunking_strategies.py", "  [语法] chunking_strategies")
if a2: try_import_local("phase_a/evaluate_chunking.py",   "  [语法] evaluate_chunking")

# 读取评测结果（如果存在）
eval_path = os.path.join(RAG_ROOT, "phase_a/results/chunking_eval.json")
if os.path.isfile(eval_path):
    with open(eval_path) as f:
        eval_data = json.load(f)
    print("\n  Phase A评测结果：")
    for strategy, metrics in eval_data.items():
        p = metrics.get("precision_at_k", metrics.get("precision_at_3", "?"))
        n = metrics.get("noise_rate", "?")
        print(f"    {strategy}: Precision@3={p:.1%}  噪声率={n:.1%}" if isinstance(p, float) else f"    {strategy}: {metrics}")

phase_a_done = all([a1, a2, a3, a4, a5])

# ============================================================
# 检查：Phase B 文件结构
# ============================================================
section("【Phase B】Embedding选型 + Chroma")
b1 = check_file("phase_b/embedding_benchmark.py", "embedding_benchmark.py")
b2 = check_file("phase_b/chroma_builder.py",      "chroma_builder.py")
b3 = check_file("phase_b/results/embedding_benchmark.json", "评测结果 embedding_benchmark.json")
b4 = check_dir("vector_store/chroma_db",           "chroma_db/（Chroma持久化目录）")

# 检查模型下载
print("\n  Embedding模型下载状态：")
MODEL_CACHE = "/root/autodl-tmp/embedding_model"
found_small = found_base = found_reranker = False
if os.path.isdir(MODEL_CACHE):
    for root, dirs, files in os.walk(MODEL_CACHE):
        if "config.json" in files:
            rel = os.path.relpath(root, MODEL_CACHE)
            size_mb = sum(
                os.path.getsize(os.path.join(root, f))
                for f in files
            ) / 1024 / 1024
            if "bge-small" in root.lower():
                ok(f"bge-small-zh  ({size_mb:.0f} MB)  {rel}")
                found_small = True
            elif "bge-base" in root.lower():
                ok(f"bge-base-zh   ({size_mb:.0f} MB)  {rel}")
                found_base = True
            elif "reranker" in root.lower():
                ok(f"bge-reranker  ({size_mb:.0f} MB)  {rel}")
                found_reranker = True
else:
    fail(f"embedding_model目录不存在: {MODEL_CACHE}")

if not found_small:   warn("bge-small-zh 未找到")
if not found_base:    warn("bge-base-zh  未找到（Phase B需要）")
if not found_reranker:warn("bge-reranker 未找到（Phase C需要）")

print()
if b1: try_import_local("phase_b/embedding_benchmark.py", "  [语法] embedding_benchmark")
if b2: try_import_local("phase_b/chroma_builder.py",      "  [语法] chroma_builder")

phase_b_done = all([b1, b2])

# ============================================================
# 检查：Phase C 文件结构
# ============================================================
section("【Phase C】企业级三件套")
c1 = check_file("phase_c/bm25_index.py",          "bm25_index.py")
c2 = check_file("phase_c/hybrid_retriever.py",    "hybrid_retriever.py")
c3 = check_file("phase_c/reranker.py",            "reranker.py")
c4 = check_file("phase_c/citation_formatter.py",  "citation_formatter.py")
c5 = check_file("phase_c/enterprise_pipeline.py", "enterprise_pipeline.py")
c6 = check_file("vector_store/bm25_index.pkl",    "BM25索引 bm25_index.pkl")
c7 = check_file("phase_c/results/enterprise_pipeline_test.json", "Pipeline测试结果")

print()
if c1: try_import_local("phase_c/bm25_index.py",         "  [语法] bm25_index")
if c2: try_import_local("phase_c/hybrid_retriever.py",   "  [语法] hybrid_retriever")
if c3: try_import_local("phase_c/reranker.py",           "  [语法] reranker")
if c4: try_import_local("phase_c/citation_formatter.py", "  [语法] citation_formatter")
if c5: try_import_local("phase_c/enterprise_pipeline.py","  [语法] enterprise_pipeline")

# 检查BM25索引完整性
bm25_path = os.path.join(RAG_ROOT, "vector_store/bm25_index.pkl")
if os.path.isfile(bm25_path):
    try:
        import pickle
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        count = len(bm25_data.get("chunks", []))
        ok(f"  BM25索引可读，包含 {count} 个文档")
    except Exception as e:
        fail(f"  BM25索引损坏: {e}")

phase_c_done = all([c1, c2, c3, c4, c5])

# ============================================================
# 检查：Phase D 文件结构
# ============================================================
section("【Phase D】消融实验")
d1 = check_file("phase_d/ablation_experiment.py",       "ablation_experiment.py")
d2 = check_file("phase_d/results/ablation_raw.json",    "消融实验原始数据")
d3 = check_file("phase_d/results/ablation_report.md",   "消融实验报告")

if d3:
    report_path = os.path.join(RAG_ROOT, "phase_d/results/ablation_report.md")
    with open(report_path) as f:
        lines = f.readlines()
    print("\n  报告预览（前15行）：")
    for line in lines[:15]:
        print(f"    {line.rstrip()}")

phase_d_done = all([d1, d2, d3])

# ============================================================
# 检查：目录结构完整性（扫描实际存在的文件）
# ============================================================
section("【实际目录扫描】rag/ 下所有 .py 和 .json 文件")
py_files   = []
json_files = []
pkl_files  = []
for dirpath, dirnames, filenames in os.walk(RAG_ROOT):
    # 跳过 __pycache__ 和 chroma_db内部
    dirnames[:] = [d for d in dirnames
                   if d not in ("__pycache__", "chroma_db", ".git")]
    for fn in sorted(filenames):
        rel = os.path.relpath(os.path.join(dirpath, fn), RAG_ROOT)
        size_kb = os.path.getsize(os.path.join(dirpath, fn)) / 1024
        if fn.endswith(".py"):
            py_files.append((rel, size_kb))
        elif fn.endswith(".json"):
            json_files.append((rel, size_kb))
        elif fn.endswith(".pkl"):
            pkl_files.append((rel, size_kb))

print("\n  .py 文件：")
for rel, size in py_files:
    print(f"    {rel:<55} {size:>7.1f} KB")

print("\n  .json 文件：")
for rel, size in json_files:
    print(f"    {rel:<55} {size:>7.1f} KB")

print("\n  .pkl 文件：")
for rel, size in pkl_files:
    print(f"    {rel:<55} {size:>7.1f} KB")

# ============================================================
# 汇总报告
# ============================================================
section("【汇总】进度概览")

status = {
    "Phase A（切分实验）":   ("✅ 已完成" if phase_a_done else "⚠️  未完成"),
    "Phase B（Embedding）":  ("✅ 已完成" if phase_b_done else "⚠️  未完成"),
    "Phase C（企业三件套）": ("✅ 已完成" if phase_c_done else "⚠️  未完成"),
    "Phase D（消融实验）":   ("✅ 已完成" if phase_d_done else "⚠️  未完成"),
}
for phase, st in status.items():
    print(f"  {st}  {phase}")

print("""
──────────────────────────────────────────────────────
下一步：把以上输出完整贴给我，
我会帮你判断：
  1. 每个Phase的具体完成度（文件存在 ≠ 正确运行）
  2. 代码改乱的具体问题点
  3. 哪些部分可以直接修复，哪些需要重新写
──────────────────────────────────────────────────────
""")