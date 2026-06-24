"""
debate.py —— 编排桥：把“历史 + 用户问题”喂给 LangGraph 图，跑完返回完整结果。

它处在 app.py(UI) 和 graph.py(图) 之间：
  - app.py 只管界面：收到用户问题 → 调 run_debate() → 把结果展示出来。
  - graph.py 只管“图怎么走”。
  - debate.py 负责：把用户问题整理成图需要的初始状态，跑图，再把完整结果交回 UI。

这样 app.py 不用关心 LangGraph 的任何细节，graph.py 也不用关心界面。

提供两条路径：
  - stream_debate(...)  异步生成器，逐个节点产出 {节点名: {字段:值}}，供 UI 边跑边推进度。
  - run_debate(...)     = 把 stream_debate 跑完合并成完整 state 返回（无 UI 场景用，如 graph.py 自测）。
"""

import config
from graph import app
from state import DebateState


def _build_initial_state(question: str, history: list[tuple[str, str]]) -> DebateState:
    """组装图的初始状态：拼好历史上下文 + 用户问题。只填这两个，其余由各节点逐步填入。"""
    history_str = _format_history(history)
    return {"question": question, "history_str": history_str}


async def stream_debate(question: str, history: list[tuple[str, str]]):
    """
    流式跑一轮圆桌辩论：每个节点跑完后立刻 yield 一个 chunk。

    yield 的格式：{节点名: {该节点更新的字段: 值}}，例如
        {'node_initial': {'initial': {...}}}
        {'node_merge':   {'final_answer': '...'}}
    （节点名来自 graph.py 的 node_initial/node_revise/node_score/node_merge。）

    为什么用流式？这样 app.py 能在每节点完成时实时给用户反馈进度，而不是全程黑盒等待。
    它自身不碰 UI，只把图的活动透传出来，职责单一。

    参数：
        question：用户这一轮的问题。
        history：历史问答 [(用户问, 最终答), ...]。只含“问+最终答案”，不含中间过程。
    """
    initial_state = _build_initial_state(question, history)
    # stream_mode="updates"：每个节点完成后产出「它更新的字段」，逐节点触发。
    async for chunk in app.astream(initial_state, stream_mode="updates"):
        yield chunk


async def run_debate(question: str, history: list[tuple[str, str]]) -> dict:
    """
    运行一轮圆桌辩论，返回完整状态（流式版的无 UI 包装）。

    参数：
        question：用户这一轮的问题。
        history：历史问答 [(用户问, 最终答), ...]。只含“问+最终答案”，不含中间过程。

    返回：完整的状态字典，含 initial / revised / scores / final_answer 等所有字段。
          app.py 据此展示全过程，主流只显示 final_answer。
    """
    # 逐个 chunk 合并成完整状态。等价于 app.ainvoke，但和流式接口共用一条路径，逻辑不重复。
    final_state: dict = _build_initial_state(question, history)
    async for chunk in stream_debate(question, history):
        for node, updates in chunk.items():
            final_state.update(updates)
    return final_state



def _format_history(history: list[tuple[str, str]]) -> str:
    """
    把历史问答列表拼成一段中文，作为各 agent 的上下文。
    只取最近 MAX_HISTORY_TURNS 轮，避免越长越爆 token。
    """
    if not history:
        return ""
    lines = ["【之前的对话历史，供参考上下文】"]
    for i, (q, a) in enumerate(history[-config.MAX_HISTORY_TURNS:], start=1):
        lines.append(f"第{i}轮——用户问：{q}\n圆桌最终答：{a}")
    return "\n\n".join(lines)
