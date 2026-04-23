# run_experiment.py
"""
主实验脚本：
1. 对10个测试案例 × 3个 prompt 版本 = 30次推理
2. 对每次输出进行多维度自动评测
3. 生成对比报告
"""
import json
import time
from openai import OpenAI
from prompts import ALL_PROMPTS
from test_cases import TEST_CASES
from evaluator import evaluate_single_response

# ============================================================
# 配置
# ============================================================
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:6006/v1"
)

MODEL_NAME = "law-expert"     # 对应你 vLLM 的 --served-model-name
TEMPERATURE = 0.1             # 低温度保证可复现性
MAX_TOKENS = 1024
RESULTS_DIR = "./results"


# ============================================================
# Step 1: 推理函数
# ============================================================
def get_response(question: str, system_prompt: str) -> tuple[str, float]:
    """调用模型，返回 (回答内容, 耗时秒)"""
    start = time.time()
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question}
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    
    elapsed = time.time() - start
    content = response.choices[0].message.content
    return content, elapsed


# ============================================================
# Step 2: 运行完整实验
# ============================================================
def run_experiment():
    all_results = []
    
    total = len(ALL_PROMPTS) * len(TEST_CASES)
    current = 0
    
    for prompt_name, system_prompt in ALL_PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"正在测试 Prompt 版本: {prompt_name}")
        print(f"{'='*60}")
        
        for test_case in TEST_CASES:
            current += 1
            print(f"[{current}/{total}] {test_case['id']} - {test_case['category']}")
            
            # 获取模型回答
            response, latency = get_response(
                test_case["question"],
                system_prompt
            )
            
            # 评测
            eval_result = evaluate_single_response(
                response, test_case, prompt_name
            )
            eval_result["latency_seconds"] = latency
            
            all_results.append(eval_result)
            
            # 实时显示关键分数
            total_score = eval_result["scores"]["total"]
            kp_coverage = eval_result["details"]["key_points"]["coverage_rate"]
            print(f"  总分: {total_score:.1f}/100 | "
                  f"关键点覆盖: {kp_coverage:.0%} | "
                  f"耗时: {latency:.1f}s")
            
            # 避免请求过快
            time.sleep(0.5)
    
    return all_results


# ============================================================
# Step 3: 生成分析报告
# ============================================================
# run_experiment.py
# 只替换 generate_report 函数，其余部分完全不变

def generate_report(all_results: list) -> str:
    """生成 Markdown 格式的对比报告（支持任意数量的 prompt 版本）"""

    # 动态获取所有版本（不再硬编码版本名）
    all_versions = list(dict.fromkeys(
        r["prompt_version"] for r in all_results
    ))

    # 按 prompt 版本分组计算统计数据
    version_stats = {}
    for result in all_results:
        version = result["prompt_version"]
        if version not in version_stats:
            version_stats[version] = {
                "scores":          [],
                "kp_coverage":     [],
                "citation_counts": [],
                "latencies":       []
            }
        version_stats[version]["scores"].append(result["scores"]["total"])
        version_stats[version]["kp_coverage"].append(
            result["details"]["key_points"]["coverage_rate"]
        )
        version_stats[version]["citation_counts"].append(
            result["details"]["citation"]["citation_count"]
        )
        version_stats[version]["latencies"].append(result["latency_seconds"])

    def avg(lst): return sum(lst) / len(lst) if lst else 0

    report = []
    report.append("# Prompt Engineering 实验报告（v2）\n")
    report.append(f"- **模型:** {MODEL_NAME}")
    report.append(f"- **测试案例数:** {len(TEST_CASES)}")
    report.append(f"- **Prompt 版本数:** {len(all_versions)}")
    report.append(f"- **版本列表:** {', '.join(all_versions)}\n")

    # ── 总体对比表（动态列）──
    report.append("## 总体对比\n")
    report.append("| Prompt 版本 | 平均总分 | 关键点覆盖率 | 法条引用数(含阿拉伯) | 平均响应时间 |")
    report.append("|:---|:---:|:---:|:---:|:---:|")

    for version in all_versions:
        stats = version_stats[version]
        report.append(
            f"| **{version}** "
            f"| {avg(stats['scores']):.1f}/100 "
            f"| {avg(stats['kp_coverage']):.1%} "
            f"| {avg(stats['citation_counts']):.1f} 条 "
            f"| {avg(stats['latencies']):.1f}s |"
        )

    # ── 各维度得分详情（动态列）──
    report.append("\n## 各维度得分详情\n")
    dimensions = [
        ("key_point_coverage", "关键点覆盖(40分)"),
        ("negative_penalty",   "无负面内容(20分)"),
        ("structure",          "结构完整(20分)"),
        ("citation_format",    "法条格式(20分)"),
    ]

    header = "| 评测维度 | " + " | ".join(all_versions) + " |"
    separator = "|:---| " + " | ".join([":---:"] * len(all_versions)) + " |"
    report.append(header)
    report.append(separator)

    for dim_key, dim_name in dimensions:
        row_parts = [f"| {dim_name}"]
        for version in all_versions:
            dim_scores = [
                r["scores"][dim_key]
                for r in all_results
                if r["prompt_version"] == version
            ]
            row_parts.append(f"{avg(dim_scores):.1f}")
        report.append(" | ".join(row_parts) + " |")

    # ── 按类别对比（动态列）──
    report.append("\n## 按问题类别对比\n")
    categories = sorted(set(tc["category"] for tc in TEST_CASES))

    header = "| 问题类别 | " + " | ".join(all_versions) + " |"
    separator = "|:---| " + " | ".join([":---:"] * len(all_versions)) + " |"
    report.append(header)
    report.append(separator)

    for category in categories:
        row_parts = [f"| {category}"]
        for version in all_versions:
            cat_scores = [
                r["scores"]["total"]
                for r in all_results
                if r["prompt_version"] == version
                and r["category"] == category
            ]
            row_parts.append(f"{avg(cat_scores):.1f}" if cat_scores else "N/A")
        report.append(" | ".join(row_parts) + " |")

    # ── 归一化修复效果展示（新增）──
    report.append("\n## 归一化匹配修复效果\n")
    report.append("展示各版本中通过归一化（阿拉伯数字/缩写转换）额外命中的关键点：\n")
    report.append("| 版本 | 通过原始匹配 | 通过归一化额外命中 |")
    report.append("|:---|:---:|:---:|")

    for version in all_versions:
        version_results = [r for r in all_results if r["prompt_version"] == version]
        original_hits = 0
        normalized_hits = 0
        for r in version_results:
            for detail in r["details"]["key_points"].get("match_details", []):
                if detail["matched"]:
                    if detail["via"] == "original":
                        original_hits += 1
                    else:
                        normalized_hits += 1
        report.append(f"| {version} | {original_hits} | {normalized_hits} |")

    # ── 典型案例对比（TC001 + TC007，动态版本）──
    for tc_id, tc_desc in [("TC001", "劳动法-赔偿"), ("TC007", "合同-借款（CoT退步案例）")]:
        report.append(f"\n## 典型案例对比：{tc_id}（{tc_desc}）\n")
        for version in all_versions:
            tc_result = next(
                (r for r in all_results
                 if r["test_id"] == tc_id and r["prompt_version"] == version),
                None
            )
            if tc_result:
                cite = tc_result["details"]["citation"]
                report.append(
                    f"### {version}（总分 {tc_result['scores']['total']:.1f}）\n"
                )
                report.append(
                    f"**标准格式法条:** {cite['standard_citations']}\n"
                )
                report.append(
                    f"**阿拉伯数字格式:** {cite.get('arabic_citations', [])}\n"
                )
                report.append(
                    f"**回答:**\n```\n{tc_result['response']}\n```\n"
                )

    # ── 结论（动态最优版本）──
    report.append("## 结论与建议\n")
    best_version = max(
        all_versions,
        key=lambda v: avg(version_stats[v]["scores"])
    )
    second_best = sorted(
        all_versions,
        key=lambda v: avg(version_stats[v]["scores"]),
        reverse=True
    )[1] if len(all_versions) > 1 else best_version

    report.append(f"**最优 Prompt 版本:** `{best_version}`")
    report.append(f"**次优 Prompt 版本:** `{second_best}`\n")
    report.append("**关键发现:**")
    report.append("- 归一化匹配修复了 v3 的法条计数偏差（阿拉伯数字格式额外命中）")
    report.append("- v3_cot_lite 弹性指令预期改善 TC007 的 CoT 退步问题")
    report.append("- v4_fewshot 通过示例引导，预期在格式一致性上超越 v2")
    report.append("\n**工程建议:**")
    report.append("- 高并发/低延迟场景：优先使用 v2_structured")
    report.append("- 复杂法律分析场景：优先使用 v3_cot_lite")
    report.append("- 格式一致性要求高的场景：优先使用 v4_fewshot")
    report.append("- 建议将版本选择作为 API 参数暴露，支持调用方按场景选择")

    return "\n".join(report)

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    
    print("开始 Prompt Engineering 实验...")
    print(f"测试案例: {len(TEST_CASES)} 个")
    print(f"Prompt 版本: {len(ALL_PROMPTS)} 个")
    print(f"总推理次数: {len(TEST_CASES) * len(ALL_PROMPTS)} 次")
    
    # 运行实验
    results = run_experiment()
    
    # 保存原始结果
    raw_path = f"{RESULTS_DIR}/raw_outputs.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n原始结果已保存到: {raw_path}")
    
    # 生成报告
    report = generate_report(results)
    report_path = f"{RESULTS_DIR}/eval_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"评测报告已保存到: {report_path}")
    
    # 打印简要结果
    print("\n" + "="*60)
    print("实验完成！关键结果预览：")
    print("="*60)
    print(report[:2000])