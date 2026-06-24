"""
chair.py —— “主席”裁判的两个能力：打分 + 合并，以及对模型返回 JSON 的健壮解析。

为什么单独拆一个文件？
主席是整条流水线里“评判者”的角色，逻辑相对独立（打分要解析 JSON、合并要排序），
放在一起便于单独阅读“裁判是怎么工作的”，也让 graph.py 更专注画图。

不懂 LangGraph 不影响读这个文件——它就是几个普通函数。
"""

import json
import re

import config
import llm
import prompts


def parse_scores(raw: str, slot_to_name: dict) -> list[dict]:
    """
    把主席返回的文本解析成打分列表。

    模型被要求只输出 JSON 数组，但现实中它偶尔会：
      - 用 ```json ... ``` 代码块包裹
      - 前后多一句“好的，这是结果：”之类的话
    所以不能直接 json.loads，要先“剥”出真正的 JSON 片段。

    参数 slot_to_name：{编号(如"选手1"): 参赛者真实name} 的映射。
    返回：[{"name","slot","score","reason"}, ...]，已按分数降序排好。
    解析彻底失败时走“降级”：给每个参赛者默认中等分，保证流程不中断。
    """
    if not raw:
        return _fallback_scores(slot_to_name, "主席未返回内容")

    # 策略1：先尝试整体解析（模型守规矩时直接成功）。
    try:
        return _normalize_scores(json.loads(raw), slot_to_name)
    except json.JSONDecodeError:
        pass

    # 策略2：用正则抓出第一个 [ 到最后一个 ] 之间的片段再解析。
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            return _normalize_scores(json.loads(match.group(0)), slot_to_name)
        except json.JSONDecodeError:
            pass

    # 都失败：降级，流程继续。
    return _fallback_scores(slot_to_name, f"主席返回无法解析：{raw[:80]}")


def _normalize_scores(data, slot_to_name: dict) -> list[dict]:
    """
    把解析出的列表规整成统一格式，并把「编号」映射回「参赛者真实 name」。
    顺便兜底：score 非数字的给默认分、漏评的补上。
    """
    result = []
    seen_slots = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("id", "")).strip()
        if slot not in slot_to_name:
            # 不认识的编号，跳过，避免脏数据（模型瞎改编号时这里挡住）
            continue
        seen_slots.add(slot)
        try:
            score = int(item.get("score", 0))
        except (TypeError, ValueError):
            score = 50
        score = max(0, min(100, score))  # 钳到 0-100
        result.append({
            "slot": slot,
            "name": slot_to_name[slot],
            "score": score,
            "reason": str(item.get("reason", "")).strip(),
        })
    # 模型漏评的参赛者补一个默认分。
    for slot, name in slot_to_name.items():
        if slot not in seen_slots:
            result.append({"slot": slot, "name": name, "score": 50, "reason": "主席未评分"})

    # 按分数降序排。
    result.sort(key=lambda x: x["score"], reverse=True)
    return result


def _fallback_scores(slot_to_name: dict, reason: str) -> list[dict]:
    """降级：解析失败时给每个参赛者默认中等分，并列出原因。第一版保证不阻断流程。"""
    return [
        {"slot": slot, "name": name, "score": 50, "reason": f"（降级）{reason}"}
        for slot, name in slot_to_name.items()
    ]


async def chair_score(question: str, revised: dict) -> list[dict]:
    """
    主席打分：让模型给所有修正回答打分，返回排好序的打分列表。
    revised: {name: 修正回答}。

    关键：内部用显式编号（选手1、选手2…）喂给主席，避免模型把中文名缩写错
    导致 id 对不上。最后再把编号映射回真实 name。
    """
    # 建「编号 → name」映射，编号用 选手1/选手2…，顺序固定。
    names = list(revised.keys())
    slot_to_name = {f"选手{i+1}": name for i, name in enumerate(names)}
    # 反过来：name → 编号。
    name_to_slot = {name: slot for slot, name in slot_to_name.items()}

    # 给主席的列表用编号展示，回答正文不变。
    revised_by_slot = [(name_to_slot[name], ans) for name, ans in revised.items()]

    raw = await llm.call_llm(
        system=prompts.SCORE_SYSTEM,
        user=prompts.score_user(question, revised_by_slot),
        temperature=config.CHAIR_TEMPERATURE,
        model=config.CHAIR.model,
        api_base=config.CHAIR.api_base,
        api_key=config.CHAIR.api_key,
    )
    return parse_scores(raw, slot_to_name)


async def chair_merge(question: str, scored: list[dict], revised: dict) -> str:
    """
    主席合并：把打分排序后的修正回答合并成最终答案。
    scored: [{"slot","name","score","reason"}, ...] 已排序（来自 chair_score）
    revised: {name: 修正回答}
    """
    # 给合并用的列表带上每个回答正文，按打分顺序排。展示用 name。
    enriched = []
    for item in scored:
        name = item["name"]
        enriched.append({
            "name": name,
            "score": item["score"],
            "reason": item.get("reason", ""),
            "answer": revised.get(name, ""),
        })
    return await llm.call_llm(
        system=prompts.MERGE_SYSTEM,
        user=prompts.merge_user(question, enriched),
        temperature=config.CHAIR_TEMPERATURE,
        model=config.CHAIR.model,
        api_base=config.CHAIR.api_base,
        api_key=config.CHAIR.api_key,
    )
