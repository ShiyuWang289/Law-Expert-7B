#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
from typing import List, Dict

def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("《", "").replace("》", "")
    s = s.replace(" ", "").replace("\n", "")
    s = s.replace("第", "").replace("条", "")
    s = s.replace("四十六", "46").replace("八十七", "87")
    return s

def law_accuracy(answer: str, key_points: List[str]) -> float:
    """法条准确率（简化版）：key_points中含‘法/条/条例’类点的命中比例"""
    law_points = [k for k in key_points if ("法" in k or "条" in k)]
    if not law_points:
        return 1.0
    ans = normalize_text(answer)
    hit = 0
    for p in law_points:
        if normalize_text(p) in ans:
            hit += 1
    return hit / len(law_points)

def keypoint_coverage(answer: str, key_points: List[str]) -> float:
    ans = normalize_text(answer)
    hit = sum(1 for p in key_points if normalize_text(p) in ans)
    return hit / max(1, len(key_points))

def repetition_rate(answer: str) -> float:
    """重复率：按句子去重粗估"""
    parts = [x.strip() for x in re.split(r"[。！？\n]", answer) if x.strip()]
    if len(parts) <= 1:
        return 0.0
    uniq = len(set(parts))
    return max(0.0, 1 - uniq / len(parts))

def avg_token_len(answer: str) -> int:
    """无 tokenizer 依赖，先用字符长度近似"""
    return len(answer)

def hallucination_count(answer: str, must_not: List[str]) -> int:
    """幻觉计数（规则版）：命中 must_not 记 1"""
    ans = normalize_text(answer)
    return sum(1 for x in must_not if normalize_text(x) in ans)

def score_case(answer: str, case: Dict) -> Dict:
    kp = case.get("key_points", [])
    mn = case.get("must_not", [])
    return {
        "law_accuracy": law_accuracy(answer, kp),
        "coverage": keypoint_coverage(answer, kp),
        "repetition_rate": repetition_rate(answer),
        "avg_len": avg_token_len(answer),
        "hallucination_count": hallucination_count(answer, mn)
    }