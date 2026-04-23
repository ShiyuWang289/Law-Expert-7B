# evaluator.py
"""
评测维度说明：
1. 关键点覆盖率  - 必须提到的法条/概念是否出现（支持等价匹配）
2. 负面内容惩罚  - 错误表述、模糊引用是否出现
3. 结构完整性    - 是否有法律分析、建议、途径三个部分
4. 法条格式规范  - 是否用了《》第X条的标准格式（含阿拉伯数字兼容）
5. 回答长度      - 字符数（太短=信息不足，太长=冗余）

v2 改进点：
- 新增阿拉伯数字 ↔ 中文数字等价匹配
- 新增法律名称缩写识别
- 法条格式检查兼容阿拉伯数字写法
- 新增 v3_cot_lite 和 v4_fewshot 的结构判断
"""
import re


# ============================================================
# 工具函数：数字与缩写等价转换
# ============================================================

# 阿拉伯数字 → 中文数字映射表（支持到199条，覆盖常见法条范围）
_ARABIC_TO_CN = {
    '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
    '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
    '10': '十', '11': '十一', '12': '十二', '13': '十三',
    '14': '十四', '15': '十五', '16': '十六', '17': '十七',
    '18': '十八', '19': '十九', '20': '二十', '21': '二十一',
    '22': '二十二', '23': '二十三', '24': '二十四', '25': '二十五',
    '26': '二十六', '27': '二十七', '28': '二十八', '29': '二十九',
    '30': '三十', '31': '三十一', '32': '三十二', '33': '三十三',
    '40': '四十', '41': '四十一', '42': '四十二', '43': '四十三',
    '44': '四十四', '45': '四十五', '46': '四十六', '47': '四十七',
    '48': '四十八', '49': '四十九', '50': '五十',
    '88': '八十八', '100': '一百', '101': '一百零一',
    '188': '一百八十八', '1024': '一千零二十四',
}

# 中文数字 → 阿拉伯数字（反向映射，自动生成）
_CN_TO_ARABIC = {v: k for k, v in _ARABIC_TO_CN.items()}

# 法律名称缩写映射表
# key = 缩写或别名，value = 标准全称（用于匹配时的归一化）
_LAW_ALIASES = {
    # 劳动合同法
    "劳动合同法":      "劳动合同法",
    "劳合法":          "劳动合同法",

    # 民法典
    "民法典":          "民法典",
    "民法通则":        "民法典",      # 旧称，归一化到民法典

    # 工伤保险条例
    "工伤保险条例":    "工伤保险条例",
    "工伤条例":        "工伤保险条例",

    # 消费者权益保护法
    "消费者权益保护法": "消费者权益保护法",
    "消法":            "消费者权益保护法",
    "消保法":          "消费者权益保护法",

    # 劳动法（旧法，与劳动合同法并存）
    "劳动法":          "劳动法",
}


def normalize_law_text(text: str) -> str:
    """
    将回答文本归一化，用于关键点匹配：
    1. 将阿拉伯数字条款号转为中文（第46条 → 第四十六条）
    2. 将法律名称缩写展开为全称
    """
    normalized = text

    # Step 1: 阿拉伯数字条款号 → 中文
    # 匹配 "第46条" 或 "第46、47条" 等格式
    def replace_arabic_article(match):
        num_str = match.group(1)
        cn = _ARABIC_TO_CN.get(num_str)
        if cn:
            return f"第{cn}条"
        return match.group(0)  # 无法转换则保持原样

    normalized = re.sub(r'第(\d+)条', replace_arabic_article, normalized)

    # Step 2: 法律名称缩写 → 标准全称
    # 只在《》书名号内做替换，避免误伤正文
    def replace_law_alias(match):
        inner = match.group(1)
        full_name = _LAW_ALIASES.get(inner, inner)
        return f"《{full_name}》"

    normalized = re.sub(r'《([^》]+)》', replace_law_alias, normalized)

    return normalized


def check_key_points(response: str, key_points: list) -> dict:
    """
    检查关键点覆盖情况。
    
    改进：先对 response 做归一化，再做字符串匹配。
    这样 "第46条" 能匹配 "第四十六条"，"消法" 能匹配 "消费者权益保护法"。
    """
    # 对回答做归一化处理
    normalized_response = normalize_law_text(response)

    covered = []
    missed = []
    match_details = []  # 记录每个关键点的匹配情况，方便调试

    for point in key_points:
        # 对关键点本身也做归一化（防止关键点里有阿拉伯数字）
        normalized_point = normalize_law_text(point)

        # 先在原始文本中找，再在归一化文本中找
        found_in_original   = point in response
        found_in_normalized = normalized_point in normalized_response

        if found_in_original or found_in_normalized:
            covered.append(point)
            match_details.append({
                "point": point,
                "matched": True,
                "via": "original" if found_in_original else "normalized"
            })
        else:
            missed.append(point)
            match_details.append({
                "point": point,
                "matched": False,
                "via": None
            })

    coverage_rate = len(covered) / len(key_points) if key_points else 1.0

    return {
        "coverage_rate": coverage_rate,
        "covered": covered,
        "missed": missed,
        "match_details": match_details,   # 新增：方便调试归一化效果
        "score": coverage_rate * 40       # 满分40分
    }


def check_negative_points(response: str, negative_points: list) -> dict:
    """检查负面内容（出现则扣分）"""
    triggered = []

    for point in negative_points:
        if point in response:
            triggered.append(point)

    penalty = len(triggered) * 5  # 每个负面内容扣5分

    return {
        "triggered": triggered,
        "penalty": penalty,
        "score": max(0, 20 - penalty)  # 满分20分
    }


def check_structure(response: str, prompt_version: str) -> dict:
    """
    检查回答结构完整性。
    
    改进：新增 v3_cot_lite 和 v4_fewshot 的判断分支。
    v3_cot_lite 使用更宽松的标记匹配（只要求部分标记出现）。
    v4_fewshot 和 v2 一样用语义结构检查（因为示例本身会引导格式）。
    """

    # ── v3_cot：严格结构标记匹配（原逻辑不变）──
    if prompt_version == "v3_cot":
        markers = ["法律定性", "法条依据", "案情分析", "行动建议"]
        found = [m for m in markers if m in response]
        score = (len(found) / len(markers)) * 20
        return {
            "type": "strict_marker",
            "found_markers": found,
            "total_markers": markers,
            "score": score
        }

    # ── v3_cot_lite：宽松结构标记匹配（新增）──
    # 只要求出现"法律分析"类 + "建议"类关键词，不要求固定标题
    if prompt_version == "v3_cot_lite":
        analysis_markers = ["法律定性", "法律分析", "法条依据", "根据", "依据"]
        suggestion_markers = ["行动建议", "具体建议", "建议", "可以", "应当"]
        path_markers = ["行动建议", "维权途径", "仲裁", "起诉", "协商", "调解"]

        has_analysis   = any(m in response for m in analysis_markers)
        has_suggestion = any(m in response for m in suggestion_markers)
        has_path       = any(m in response for m in path_markers)

        components = [has_analysis, has_suggestion, has_path]
        score = (sum(components) / len(components)) * 20
        return {
            "type": "loose_marker",
            "has_analysis":   has_analysis,
            "has_suggestion": has_suggestion,
            "has_path":       has_path,
            "score": score
        }

    # ── v4_fewshot：语义结构检查（与 v2 相同逻辑）──
    # few-shot 示例会引导格式，不需要强制标记，用语义关键词判断即可
    if prompt_version == "v4_fewshot":
        has_legal_analysis = any(kw in response for kw in
                                 ["根据", "依据", "法律规定", "条规定"])
        has_suggestion     = any(kw in response for kw in
                                 ["建议", "可以", "应当", "需要"])
        has_action_path    = any(kw in response for kw in
                                 ["仲裁", "起诉", "投诉", "协商", "调解"])

        components = [has_legal_analysis, has_suggestion, has_action_path]
        score = (sum(components) / len(components)) * 20
        return {
            "type": "semantic",
            "has_legal_analysis": has_legal_analysis,
            "has_suggestion":     has_suggestion,
            "has_action_path":    has_action_path,
            "score": score
        }

    # ── v1_baseline / v2_structured：语义结构检查（原逻辑）──
    has_legal_analysis = any(kw in response for kw in
                             ["根据", "依据", "法律规定", "条规定"])
    has_suggestion     = any(kw in response for kw in
                             ["建议", "可以", "应当", "需要"])
    has_action_path    = any(kw in response for kw in
                             ["仲裁", "起诉", "投诉", "协商", "调解"])

    components = [has_legal_analysis, has_suggestion, has_action_path]
    score = (sum(components) / len(components)) * 20

    return {
        "type": "semantic",
        "has_legal_analysis": has_legal_analysis,
        "has_suggestion":     has_suggestion,
        "has_action_path":    has_action_path,
        "score": score
    }


def check_law_citation_format(response: str) -> dict:
    """
    检查法条引用格式规范性。
    
    改进：
    1. 阿拉伯数字格式（《劳动合同法》第46条）也计入有效引用
    2. 对两种格式分别统计，在报告中区分展示
    3. 归一化后的引用总数作为 citation_count（更准确反映引用意愿）
    """

    # ── 标准格式：《法律名称》第X条（中文数字）──
    standard_pattern = r'《[^》]+》第[零一二三四五六七八九十百千]+条'
    standard_citations = re.findall(standard_pattern, response)

    # ── 兼容格式：《法律名称》第N条（阿拉伯数字）──
    # 改进点：原来只统计但不计分，现在也给分
    arabic_pattern = r'《[^》]+》第\d+条'
    arabic_citations = re.findall(arabic_pattern, response)

    # ── 模糊引用（扣分项）──
    vague_pattern = r'(相关法律|有关规定|依据法律|相关规定)'
    vague_citations = re.findall(vague_pattern, response)

    # ── 计分逻辑 ──
    # 标准格式：每条 5 分；阿拉伯数字格式：每条 3 分（鼓励但略低于标准）
    # 上限 20 分；模糊引用：每次 -3 分
    valid_count = len(standard_citations) + len(arabic_citations)
    raw_score = len(standard_citations) * 5 + len(arabic_citations) * 3
    score = min(20, raw_score)
    score -= len(vague_citations) * 3
    score = max(0, score)

    return {
        "standard_citations": standard_citations,      # 中文数字格式
        "arabic_citations":   arabic_citations,        # 阿拉伯数字格式（新增）
        "vague_citations":    vague_citations,
        "citation_count":     valid_count,             # 两种格式合计（新增）
        "standard_count":     len(standard_citations), # 新增，方便对比
        "score": score
    }


def check_length(response: str) -> dict:
    """检查回答长度合理性"""
    length = len(response)

    if length < 100:
        score, comment = 0,  "回答过短"
    elif length < 200:
        score, comment = 5,  "回答偏短"
    elif length < 800:
        score, comment = 10, "长度合理"
    elif length < 1500:
        score, comment = 8,  "回答稍长"
    else:
        score, comment = 5,  "回答过长，注意是否有重复内容"

    return {
        "length":  length,
        "score":   score,
        "comment": comment
    }


def evaluate_single_response(
    response: str,
    test_case: dict,
    prompt_version: str
) -> dict:
    """对单个回答进行全维度评测"""

    kp_result     = check_key_points(response, test_case["key_points"])
    neg_result    = check_negative_points(response, test_case["negative_points"])
    struct_result = check_structure(response, prompt_version)
    cite_result   = check_law_citation_format(response)
    len_result    = check_length(response)

    total_score = (
        kp_result["score"]     +   # 40分
        neg_result["score"]    +   # 20分
        struct_result["score"] +   # 20分
        cite_result["score"]   +   # 20分
        len_result["score"]        # 10分（超出部分被 min(100) 截断）
    )
    total_score = min(100, total_score)

    return {
        "test_id":        test_case["id"],
        "category":       test_case["category"],
        "prompt_version": prompt_version,
        "scores": {
            "key_point_coverage": kp_result["score"],
            "negative_penalty":   neg_result["score"],
            "structure":          struct_result["score"],
            "citation_format":    cite_result["score"],
            "length":             len_result["score"],
            "total":              total_score
        },
        "details": {
            "key_points": kp_result,
            "negative":   neg_result,
            "structure":  struct_result,
            "citation":   cite_result,
            "length":     len_result
        },
        "response": response
    }