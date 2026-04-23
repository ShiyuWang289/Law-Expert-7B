import json
import os
import subprocess
import sys
import re

# ========================================
# 基础评估结果（来自 results.log）
# ========================================
basic_results = {
    "base": {
        "Average": 78.83,
        "STEM": 72.79,
        "Social Sciences": 85.82,
        "Humanities": 78.21,
        "Other": 80.99
    },
    "sft": {
        "Average": 79.42,
        "STEM": 73.26,
        "Social Sciences": 86.18,
        "Humanities": 79.38,
        "Other": 81.51
    },
    "dpo": {
        "Average": 79.05,
        "STEM": 73.72,
        "Social Sciences": 85.82,
        "Humanities": 78.60,
        "Other": 80.47
    }
}

# ========================================
# 第一步：修复 domain_eval.py 中的模型路径问题
# ========================================
def fix_domain_eval_py():
    """修复 domain_eval.py 中模型名称中的下划线问题"""
    try:
        with open("domain_eval.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # 检查是否需要修复
        if "Qwen2___5-7B" in content or "replace" not in content:
            print("🔧 修复 domain_eval.py 中的模型路径问题...")
            
            # 替换模型路径中的下划线处理
            # 原来: model_path = ... 
            # 新的: model_path = model_path.replace('___', '_')
            
            modified = False
            
            # 在 AutoTokenizer.from_pretrained 之前添加修复代码
            if "model_path.replace" not in content:
                # 找到第一个 AutoTokenizer.from_pretrained 位置，在其前面添加修复代码
                pattern = r"(tokenizer = AutoTokenizer\.from_pretrained\(model_path)"
                replacement = r"# 修复模型路径中的下划线问题\n    model_path = model_path.replace('___', '_')\n    \1"
                new_content = re.sub(pattern, replacement, content)
                
                if new_content != content:
                    modified = True
                    content = new_content
            
            # 同时对 AutoModelForCausalLM 做相同修复
            if "model_path.replace" in content:
                pattern = r"(model = AutoModelForCausalLM\.from_pretrained\(model_path)"
                # 只替换还没有修复的
                if not re.search(r"model_path = model_path\.replace.*?\n\s*model = AutoModelForCausalLM", content, re.DOTALL):
                    replacement = r"# 修复模型路径中的下划线问题\n    model_path = model_path.replace('___', '_')\n    \1"
                    new_content = re.sub(pattern, replacement, content)
                    if new_content != content:
                        modified = True
                        content = new_content
            
            if modified:
                with open("domain_eval.py", "w", encoding="utf-8") as f:
                    f.write(content)
                print("✅ domain_eval.py 已修复")
                return True
    except Exception as e:
        print(f"⚠️  修复 domain_eval.py 失败: {e}")
    
    return False

# 尝试修复
fix_domain_eval_py()

# ========================================
# 第二步：运行 domain_eval.py 生成法律领域评测结果
# ========================================
models = ["base", "sft", "dpo"]
print("\n" + "="*80)
print("🔄 正在生成法律领域专项评估结果...")
print("="*80)

for model in models:
    print(f"\n📊 正在评估 {model.upper()} 模型...")
    try:
        result = subprocess.run(
            [sys.executable, "domain_eval.py", "--model", model],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"⚠️  {model} 模型评估出错:")
            # 只打印错误的关键部分
            stderr = result.stderr
            if "HFValidationError" in stderr:
                print("   模型路径包含特殊字符，尝试使用环境变量方式...")
                # 使用环境变量方式重试
                env = os.environ.copy()
                env["TRANSFORMERS_OFFLINE"] = "1"
                result = subprocess.run(
                    [sys.executable, "domain_eval.py", "--model", model],
                    capture_output=True,
                    text=True,
                    timeout=600,
                    env=env
                )
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"   仍然失败，跳过 {model} 模型")
            else:
                print(f"   {stderr[:300]}")
    except subprocess.TimeoutExpired:
        print(f"❌ {model} 模型评估超时（>10分钟）")
    except FileNotFoundError:
        print(f"⚠️  未找到 domain_eval.py 文件")
        break

print("\n" + "="*80)
print("✅ 法律领域评估完成！")
print("="*80)

# ========================================
# 第三步：读取法律领域评测结果
# ========================================
domain_results = {}
available_models = []
for model in models:
    path = f"eval_results/domain_eval_{model}.json"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 检查数据结构：如果是列表就直接用，如果是字典就取 results 字段
            if isinstance(data, list):
                domain_results[model] = data
            elif isinstance(data, dict) and "results" in data:
                domain_results[model] = data["results"]
            else:
                domain_results[model] = data
            available_models.append(model)
            print(f"✅ 已读取: {path} ({len(domain_results[model])} 条记录)")
    except FileNotFoundError:
        print(f"⚠️  未找到: {path}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 格式错误: {path} - {e}")

# ========================================
# 打印基础评测结果对比表
# ========================================
print("\n" + "="*90)
print("📊 模型评测结果对比 — 各领域准确率")
print("="*90)
print(f"{'指标':<15} {'Base':>10} {'SFT':>10} {'DPO':>10} {'SFT提升':>12} {'DPO提升':>12}")
print("-"*90)

domains = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
for domain in domains:
    base = basic_results["base"][domain]
    sft = basic_results["sft"][domain]
    dpo = basic_results["dpo"][domain]
    sft_delta = sft - base
    dpo_delta = dpo - base
    
    sft_delta_str = f"+{sft_delta:.2f}" if sft_delta >= 0 else f"{sft_delta:.2f}"
    dpo_delta_str = f"+{dpo_delta:.2f}" if dpo_delta >= 0 else f"{dpo_delta:.2f}"
    
    print(f"{domain:<15} {base:>10.2f} {sft:>10.2f} {dpo:>10.2f} {sft_delta_str:>12} {dpo_delta_str:>12}")

print("="*90)

# ========================================
# 打印法律领域专项评估对比表
# ========================================
if domain_results and len(available_models) > 0:
    print("\n" + "="*130)
    print("⚖️  法律领域专项评估 — 关键点覆盖率对比")
    print("="*130)
    
    if len(available_models) == 3:
        print(f"{'Case ID':<18} {'类别':<14} {'描述':<18} {'Base':>8} {'SFT':>8} {'DPO':>8} {'SFT提升':>10} {'DPO提升':>10}")
    elif len(available_models) == 2:
        print(f"{'Case ID':<18} {'类别':<14} {'描述':<18} {available_models[0].upper():>8} {available_models[1].upper():>8} {available_models[1].upper()+'提升':>10}")
    else:
        print(f"{'Case ID':<18} {'类别':<14} {'描述':<18} {available_models[0].upper():>15}")
    
    print("-"*130)
    
    # 使用可用的模型数据
    first_model = available_models[0]
    cases = domain_results.get(first_model, [])
    
    for i, case in enumerate(cases):
        case_id = case.get("id", f"Case_{i+1}")
        category = case.get("category", "未分类")
        description = case.get("description", "")[:16]
        
        if len(available_models) == 3:
            base_cov = domain_results["base"][i].get("coverage_rate", 0) * 100
            sft_cov = domain_results["sft"][i].get("coverage_rate", 0) * 100
            dpo_cov = domain_results["dpo"][i].get("coverage_rate", 0) * 100
            sft_delta = sft_cov - base_cov
            dpo_delta = dpo_cov - base_cov
            sft_delta_str = f"+{sft_delta:.1f}%" if sft_delta >= 0 else f"{sft_delta:.1f}%"
            dpo_delta_str = f"+{dpo_delta:.1f}%" if dpo_delta >= 0 else f"{dpo_delta:.1f}%"
            
            print(f"  {case_id:<16} {category:<14} {description:<18} {base_cov:>7.1f}% {sft_cov:>7.1f}% {dpo_cov:>7.1f}% {sft_delta_str:>10} {dpo_delta_str:>10}")
        
        elif len(available_models) == 2:
            cov1 = domain_results[available_models[0]][i].get("coverage_rate", 0) * 100
            cov2 = domain_results[available_models[1]][i].get("coverage_rate", 0) * 100
            delta = cov2 - cov1
            delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            
            print(f"  {case_id:<16} {category:<14} {description:<18} {cov1:>7.1f}% {cov2:>7.1f}% {delta_str:>10}")
        
        else:
            cov = domain_results[available_models[0]][i].get("coverage_rate", 0) * 100
            print(f"  {case_id:<16} {category:<14} {description:<18} {cov:>15.1f}%")
    
    print("-"*130)
    
    # 计算平均覆盖率
    if len(available_models) == 3:
        avg_base_cov = sum(c.get("coverage_rate", 0) for c in domain_results["base"]) / len(domain_results["base"]) * 100 if domain_results["base"] else 0
        avg_sft_cov = sum(c.get("coverage_rate", 0) for c in domain_results["sft"]) / len(domain_results["sft"]) * 100 if domain_results["sft"] else 0
        avg_dpo_cov = sum(c.get("coverage_rate", 0) for c in domain_results["dpo"]) / len(domain_results["dpo"]) * 100 if domain_results["dpo"] else 0
        avg_sft_delta = avg_sft_cov - avg_base_cov
        avg_dpo_delta = avg_dpo_cov - avg_base_cov
        avg_sft_delta_str = f"+{avg_sft_delta:.1f}%" if avg_sft_delta >= 0 else f"{avg_sft_delta:.1f}%"
        avg_dpo_delta_str = f"+{avg_dpo_delta:.1f}%" if avg_dpo_delta >= 0 else f"{avg_dpo_delta:.1f}%"
        
        print(f"  {'平均值':<16} {'':14} {'':18} {avg_base_cov:>7.1f}% {avg_sft_cov:>7.1f}% {avg_dpo_cov:>7.1f}% {avg_sft_delta_str:>10} {avg_dpo_delta_str:>10}")
    
    elif len(available_models) == 2:
        avg_cov1 = sum(c.get("coverage_rate", 0) for c in domain_results[available_models[0]]) / len(domain_results[available_models[0]]) * 100
        avg_cov2 = sum(c.get("coverage_rate", 0) for c in domain_results[available_models[1]]) / len(domain_results[available_models[1]]) * 100
        avg_delta = avg_cov2 - avg_cov1
        avg_delta_str = f"+{avg_delta:.1f}%" if avg_delta >= 0 else f"{avg_delta:.1f}%"
        
        print(f"  {'平均值':<16} {'':14} {'':18} {avg_cov1:>7.1f}% {avg_cov2:>7.1f}% {avg_delta_str:>10}")
    
    print("="*130)
    
    # ========================================
    # 打印法律领域详细分析 — 覆盖的关键点
    # ========================================
    print("\n" + "="*130)
    print("⚖️  法律领域专项评估 — 覆盖的关键法律条款")
    print("="*130)
    
    if len(available_models) == 3:
        print(f"{'Case ID':<18} {'类别':<14} {'Base 覆盖':<32} {'SFT 覆盖':<32} {'DPO 覆盖':<32}")
    elif len(available_models) == 2:
        print(f"{'Case ID':<18} {'类别':<14} {available_models[0].upper()+' 覆盖':<40} {available_models[1].upper()+' 覆盖':<40}")
    else:
        print(f"{'Case ID':<18} {'类别':<14} {available_models[0].upper()+' 覆盖':<70}")
    
    print("-"*130)
    
    for i, case in enumerate(cases):
        case_id = case.get("id", f"Case_{i+1}")
        category = case.get("category", "未分类")
        
        if len(available_models) == 3:
            # 展示三个模型的关键点
            for model in ["base", "sft", "dpo"]:
                points = domain_results[model][i].get("covered_points", [])
                points_str = ", ".join(points[:2]) if points else "无"
                if len(points) > 2:
                    points_str += f" 等({len(points)})"
                
                if model == "base":
                    base_str = points_str
                elif model == "sft":
                    sft_str = points_str
                else:
                    dpo_str = points_str
            
            print(f"  {case_id:<16} {category:<14} {base_str:<32} {sft_str:<32} {dpo_str:<32}")
        
        elif len(available_models) == 2:
            points1 = domain_results[available_models[0]][i].get("covered_points", [])
            points1_str = ", ".join(points1[:2]) if points1 else "无"
            if len(points1) > 2:
                points1_str += f" 等({len(points1)})"
            
            points2 = domain_results[available_models[1]][i].get("covered_points", [])
            points2_str = ", ".join(points2[:2]) if points2 else "无"
            if len(points2) > 2:
                points2_str += f" 等({len(points2)})"
            
            print(f"  {case_id:<16} {category:<14} {points1_str:<40} {points2_str:<40}")
        
        else:
            points = domain_results[available_models[0]][i].get("covered_points", [])
            points_str = ", ".join(points) if points else "无"
            print(f"  {case_id:<16} {category:<14} {points_str:<70}")
    
    print("="*130)
    
    # ========================================
    # 按类别统计法律领域性能
    # ========================================
    print("\n" + "="*120)
    print("⚖️  法律领域专项评估 — 按类别统计")
    print("="*120)
    
    categories = {}
    for case in cases:
        cat = case.get("category", "其他")
        if cat not in categories:
            categories[cat] = {"count": 0}
            for model in available_models:
                categories[cat][model] = 0
        categories[cat]["count"] += 1
    
    for model in available_models:
        for i, case in enumerate(domain_results[model]):
            cat = case.get("category", "其他")
            if cat in categories:
                categories[cat][model] += case.get("coverage_rate", 0)
    
    header = f"{'法律类别':<16} {'案例数':>8}"
    for model in available_models:
        header += f" {model.upper()+'覆盖':>12}"
    if len(available_models) == 3:
        header += f" {'SFT提升':>12} {'DPO提升':>12}"
    elif len(available_models) == 2:
        header += f" {available_models[1].upper()+'提升':>12}"
    
    print(header)
    print("-"*120)
    
    for cat in sorted(categories.keys()):
        data = categories[cat]
        line = f"  {cat:<14} {data['count']:>8}"
        
        for model in available_models:
            avg = (data[model] / data["count"] * 100) if data["count"] > 0 else 0
            line += f" {avg:>11.1f}%"
        
        if len(available_models) == 3:
            avg_base = (data["base"] / data["count"] * 100) if data["count"] > 0 else 0
            avg_sft = (data["sft"] / data["count"] * 100) if data["count"] > 0 else 0
            avg_dpo = (data["dpo"] / data["count"] * 100) if data["count"] > 0 else 0
            delta_sft = avg_sft - avg_base
            delta_dpo = avg_dpo - avg_base
            delta_sft_str = f"+{delta_sft:.1f}%" if delta_sft >= 0 else f"{delta_sft:.1f}%"
            delta_dpo_str = f"+{delta_dpo:.1f}%" if delta_dpo >= 0 else f"{delta_dpo:.1f}%"
            line += f" {delta_sft_str:>12} {delta_dpo_str:>12}"
        elif len(available_models) == 2:
            avg1 = (data[available_models[0]] / data["count"] * 100) if data["count"] > 0 else 0
            avg2 = (data[available_models[1]] / data["count"] * 100) if data["count"] > 0 else 0
            delta = avg2 - avg1
            delta_str = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"
            line += f" {delta_str:>12}"
        
        print(line)
    
    print("="*120)
    
else:
    print("\n⚠️  未找到法律领域评测数据。")
    if domain_results:
        print(f"   已读取的模型: {available_models}")

print("\n✅ 报告生成完毕！")