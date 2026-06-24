"""
prompts.py —— 4 个辩论阶段对应的中文提示词模板。

每个函数返回一段拼好的字符串，作为给模型 system / user 的内容。
把提示词集中在这里，方便单独阅读和调优，改文风、加要求都来这里。

4 个阶段对应 graph.py 的 4 个节点：
  ① initial   各自作答
  ② revise    各自修正(参考他人回答后)
  ③ score     主席打分
  ④ merge     主席合并

(曾经有过 ② review 两两互评 阶段，已并入 ② revise ——模型直接看他人原始回答来修正，
 比看二手评价更全更准、调用也更少。)
"""

import config


# ──────────────────────────────────────────────
# 公共：把历史问答拼成中文上下文段落
# ──────────────────────────────────────────────

def history_block(history: list[tuple[str, str]]) -> str:
    """
    把历史问答列表拼成一段中文，塞进提示词作为上下文。

    history 是 [(用户问, 最终答), ...] 的列表，只存“问 + 最终答案”，
    不存中间过程（见 CLAUDE.md 的会话上下文规则）。
    没有历史就返回空串，调用方据此决定要不要加这一段。
    """
    if not history:
        return ""
    lines = ["【之前的对话历史，供你参考上下文，不要重复其中的内容】"]
    for i, (q, a) in enumerate(history[-config.MAX_HISTORY_TURNS:], start=1):
        lines.append(f"第{i}轮：\n用户问：{q}\n圆桌最终答：{a}")
    return "\n\n".join(lines) + "\n\n"


# ──────────────────────────────────────────────
# ① 各自作答
# ──────────────────────────────────────────────

def initial_system(persona: str, history: list[tuple[str, str]]) -> str:
    """第①阶段的 system 提示词：告知角色人设 + 历史上下文。"""
    hist = history_block(history)
    return (
        f"{persona}\n\n"
        "你正参与一场「圆桌」多模型辩论。现在请先凭自己的理解，独立回答用户的问题。"
        "这是第一轮作答，之后你还会看到其他参与者的回答并有机会据此修正，所以这一轮尽情给出你认为好的答案即可。\n\n"
        f"{hist}"
    )


def initial_user(question: str) -> str:
    """第①阶段的 user 内容：就是用户的问题本身。"""
    return f"用户问题：{question}"


# ──────────────────────────────────────────────
# ② 各自修正（参考他人回答后直接修正）
# ──────────────────────────────────────────────

def revise_system(persona: str, history: list[tuple[str, str]]) -> str:
    """第②阶段的 system 提示词：看到其他人的初始回答后，参考并修正自己的回答。"""
    hist = history_block(history)
    return (
        f"{persona}\n\n"
        "你正参与一场「圆桌」多模型辩论。现在进入「参考他人、修正自己」环节。\n"
        "你会看到：用户的问题、你自己的初始回答，以及其他参与者(匿名)的初始回答。\n"
        "请参考其他人的回答：吸收他们有而你没有的好观点，纠正自己回答里的错误或缺漏。"
        "但要保持独立判断——别人说得不对的不要盲从，也不要简单照抄他人原文，"
        "给出一个更完善的最终版本。\n\n"
        f"{hist}"
    )


def revise_user(question: str, my_answer: str, others: list[tuple[str, str]] | None) -> str:
    """
    第②阶段的 user 内容：问题 + 自己初始回答 + 其他参与者(匿名)的回答。

    others 是 [(编号, 回答), ...]，编号形如「选手1」，由调用方(graph.py 的 node_revise)
    排除掉「自己」后按固定顺序局部编号，既保证匿名、又不暴露他人身份。
    空列表(理论只在 1 选手时出现)给兜底文案。
    """
    if others:
        others_text = "\n\n---\n\n".join(f"=== {slot} 的回答 ===\n{ans}" for slot, ans in others)
    else:
        others_text = "(本轮只有你一位参与者，暂无他人回答可参考)"
    return (
        f"用户问题：{question}\n\n"
        f"=== 你的初始回答 ===\n{my_answer}\n\n"
        f"=== 其他参与者的回答 ===\n{others_text}\n\n"
        "请在参考他人回答后，给出你修正后的最终回答。直接输出回答正文，不要加多余说明。"
    )


# ──────────────────────────────────────────────
# ③ 主席打分
# ──────────────────────────────────────────────

SCORE_SYSTEM = (
    "你是「圆桌」辩论的主席(裁判)。现在需要对各参与者修正后的回答进行打分排序。\n"
    "评分维度：准确性、完整性、清晰度、对问题的契合度。每项满分 25，合计满分 100。\n\n"
    "你必须只输出一个 JSON 数组，不要输出任何其它文字、不要用 markdown 代码块包裹。\n"
    "格式严格如下：\n"
    '[{"id": "选手1", "score": 88, "reason": "简短点评"}, ...]\n'
    "注意：id 必须【原样使用】我们给你的参与者编号（形如 选手1、选手2…），"
    "不得改写、缩写或自行编号。score 是 0-100 的整数，reason 是一句话点评。"
)


def score_user(question: str, revised: list) -> str:
    """
    第③阶段的 user 内容：问题 + 所有修正回答(带显式编号)。
    revised 是 [(编号, 修正回答), ...] 的列表，编号形如「选手1」。
    用显式数字编号而不是参赛者名字，避免模型把名字缩写错导致 id 对不上。
    """
    parts = [f"用户问题：{question}\n\n以下是各参与者修正后的回答，请逐个打分："]
    for slot, answer in revised:
        parts.append(f"=== {slot} 的回答 ===\n{answer}")
    parts.append('\n请输出 JSON 数组，id 必须原样用上面的编号，例如：[{"id":"选手1","score":90,"reason":"..."}]')
    return "\n\n".join(parts)


# ──────────────────────────────────────────────
# ④ 主席合并
# ──────────────────────────────────────────────

MERGE_SYSTEM = (
    "你是「圆桌」辩论的主席。各参与者已经过「作答→参考他人回答修正」并打分排序。\n"
    "现在请你把这些修正回答合并、提炼成一份最终答案交给用户。\n"
    "要求：博采众长，取各回答的优点，纠正其中的错误，消除重复；"
    "用连贯的中文输出，结构清晰；直接给最终答案正文，不要介绍过程。"
)


def merge_user(question: str, scored: list[dict]) -> str:
    """
    第④阶段的 user 内容：问题 + 按分数从高到低的修正回答(含分数与点评)。
    scored 是 [{"slot","name","score","reason","answer"}, ...]，调用前应已按 score 降序排好。
    展示用 name，内部追踪用 slot(显式编号)。
    """
    parts = [f"用户问题：{question}\n\n以下是按分数从高到低排列的各修正回答："]
    for item in scored:
        parts.append(
            f"=== {item['name']}（分数 {item['score']}，点评：{item.get('reason','')}）===\n"
            f"{item['answer']}"
        )
    parts.append("\n请综合以上内容，输出一份最终答案。")
    return "\n\n".join(parts)
