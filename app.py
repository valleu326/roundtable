"""
app.py —— Chainlit 前端入口。

不懂 Chainlit 的话，先理解它是个“事件驱动”的聊天框架：
- 用户在网页发消息 → 触发 @cl.on_message 装饰的函数。
- 你在里面用 cl.Message / cl.Step 把内容发回网页。

本文件做两件事：
1. 展示“最终答案”在主对话流里（用户一眼看到的）。
2. 辩论过程中，用 cl.Step 把 LangGraph 的 4 个阶段进度实时推给用户
   （哪个阶段在跑、哪个已完成、卡在哪一步），避免全程黑盒等待。
   原始回答/评分不再单独设折叠面板 —— 进度流已覆盖全过程。

会话历史：用 cl.user_session 存 [(问, 最终答), ...]。
只有“问 + 最终答案”进历史，中间过程不进（见 CLAUDE.md 的上下文规则）。
"""

import chainlit as cl

import config
from debate import stream_debate


def _render_answers(answers: dict) -> str:
    """把 {成员名: 回答} 渲染成 Markdown 串，各成员之间用分隔线隔开。用于初始/修正回答。"""
    if not answers:
        return "（本步未产出内容）"
    blocks = [f"**{name}：**\n{ans}" for name, ans in answers.items()]
    return "\n\n---\n\n".join(blocks)


def _render_scores(scores: list) -> str:
    """把主席评分列表渲染成「排名|成员|分数|点评」表格。"""
    if not scores:
        return "（本步未产出内容）"
    lines = ["| 排名 | 成员 | 分数 | 点评 |", "|---|---|---|---|"]
    for rank, s in enumerate(scores, start=1):
        reason = s.get("reason", "").replace("|", "\\|")
        name = s.get("name", s.get("slot", ""))
        lines.append(f"| {rank} | {name} | {s.get('score', '')} | {reason} |")
    return "\n".join(lines)


def _stage_content(node_name: str, final_state: dict) -> str:
    """按节点名返回该阶段应写进步骤正文的内容；④ 合并不写正文（最终答案走主消息）。"""
    if node_name == "node_initial":
        return _render_answers(final_state.get("initial", {}))
    if node_name == "node_revise":
        return _render_answers(final_state.get("revised", {}))
    if node_name == "node_score":
        return _render_scores(final_state.get("scores", []))
    return ""  # node_merge：最终答案已作为主消息发出，步骤正文留空


# agent id → 中文名 的映射，UI 展示用。
@cl.on_chat_start
async def on_chat_start():
    """
    每次新开一个对话窗口时触发一次。
    做两件事：① 初始化空的会话历史；② 发一句欢迎语 + 配置提示。
    """
    # user_session 是 Chainlit 提供的“按会话隔离”的存储，
    # 不同用户的对话互不干扰。这里用来存历史问答列表。
    cl.user_session.set("history", [])

    # 检查配置是否就绪（选手是否够、key 是否填了）。
    missing = config.check_ready()
    if missing:
        lines = ["⚠️ 圆桌配置未就绪，请打开 config.yaml 检查："]
        for m in missing:
            lines.append(f"  - {m}")
        lines.append("\n改好 config.yaml 后重启即可，无需其它配置。")
        await cl.Message(content="\n".join(lines)).send()
        return

    await cl.Message(
        content=(
            "🪑 **欢迎来到「圆桌 RoundTable」**\n\n"
            f"现在有 {len(config.AGENTS)} 位 AI 成员围坐圆桌：" +
            "、".join(a.name for a in config.AGENTS) +
            "。\n\n你提一个问题，它们会各自作答、互相评价、据此修正，"
            "最后由主席打分排序、合并出一份最终答案。\n\n"
            "（辩论过程默认折叠在每条回答下，点开即可查看。）"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    每次用户发消息时触发。这是整个前端的核心。
    流程：取历史 → 流式跑一轮辩论、边跑边亮进度 → 主消息给最终答案 → 更新历史。
    """
    question = message.content
    history: list[tuple[str, str]] = cl.user_session.get("history") or []

    missing = config.check_ready()
    if missing:
        await cl.Message(
            content="⚠️ 配置未就绪：" + "；".join(missing) + "\n请改好 config.yaml 后重启。"
        ).send()
        return

    # 4 个阶段的进度定义：(图节点名, 进行中文案, 完成文案, 默认展开?)。
    # 节点名来自 graph.py（函数名即节点名）；文案按「① ② ③ ④」对应 4 阶段。
    # 展开策略：①②(长文本)全程折叠，避免内容写入时「展开再折叠」的跳变；
    #          ③ 评分表始终展开，让用户一眼看到结果；④ 合并无正文，折叠即可。
    n = len(config.AGENTS)
    stages = [
        ("node_initial", f"⏳ ① {n} 位成员各自作答中…", "✅ ① 初始作答完成", False),
        ("node_revise",  "⏳ ② 成员们参考他人回答修正中…", "✅ ② 各自修正完成", False),
        ("node_score",   "⏳ ③ 主席打分排序中…", "✅ ③ 主席评分完成", True),
        ("node_merge",   "⏳ ④ 主席合并最终答案中…", "✅ ④ 主席合并完成", False),
    ]
    # 节点名 -> 阶段下标，方便按 chunk 的 key 查到是哪一步。
    stage_index = {name: i for i, (name, *_rest) in enumerate(stages)}

    final_state: dict = {}
    active_step: cl.Step | None = None  # 当前「进行中」的子 Step，用于异常时标红

    # 顶层进度容器，default_open=True 让子进度默认可见。
    async with cl.Step(type="run", name="🪑 圆桌辩论进行中", default_open=True):
        try:
            # 先把第 1 阶段开为「进行中」。LangGraph 的 updates 是「节点完成后产出」，
            # 所以「进行中」靠我们先建出来，等对应 chunk 到达时再改成「✅ 完成」。
            # default_open 取自该阶段的 open 标志；auto_collapse 取反——展开的阶段
            # 不许自动收起(评分表要常驻可见)，折叠的阶段允许写完内容后自动收起。
            _, wip0, _, open0 = stages[0]
            active_step = cl.Step(type="tool", name=wip0, default_open=open0, auto_collapse=not open0)
            await active_step.__aenter__()

            name_to_step = {stages[0][0]: active_step}

            async for chunk in stream_debate(question, history):
                # chunk 形如 {节点名: {字段: 值}}。通常只有一个节点，取它出来。
                node_name = next(iter(chunk))
                final_state.update(chunk[node_name])

                # 该节点跑完 → 把它的 Step 改名为「✅ 完成」。
                idx = stage_index[node_name]
                step = name_to_step[node_name]
                step.name = stages[idx][2]
                # 首次写正文：执行中没塞过正文，这里用实际产出顶上（状态在 name 里，
                # 内容在正文里，各管各的，无需清空旧文本）。
                content = _stage_content(node_name, final_state)
                if content:
                    await step.stream_token(content)
                await step.update()
                await step.__aexit__(None, None, None)

                # 不是最后阶段 → 立刻开下一阶段为「进行中」，让用户看到进度往前走。
                if idx + 1 < len(stages):
                    _, next_wip, _, next_open = stages[idx + 1]
                    nxt = cl.Step(type="tool", name=next_wip, default_open=next_open, auto_collapse=not next_open)
                    await nxt.__aenter__()
                    name_to_step[stages[idx + 1][0]] = nxt
                    active_step = nxt
                else:
                    active_step = None

        except Exception as err:  # noqa: BLE001 —— 节点异常统一捕获，给友好提示
            # 把当前「进行中」的 Step 改成「❌ 卡在：…」，让用户看到停在哪一步。
            if active_step is not None:
                try:
                    active_step.name = "❌ 卡在这一步，详见下方提示"
                    await active_step.update()
                except Exception:  # noqa: BLE001
                    pass
                active_step = None
            await cl.Message(
                content=(
                    f"❌ 辩论过程出错：{err}\n"
                    "（多半是网络或模型限流，请稍后重试。）"
                )
            ).send()
            return

    # ── 主对话流：最终答案 ──
    final_answer = final_state.get("final_answer", "(本轮未能产出最终答案)")
    await cl.Message(content=f"🎙️ **圆桌最终答案**\n\n{final_answer}").send()

    # ── 更新会话历史：只存“问 + 最终答案”，不存中间过程 ──
    history.append((question, final_answer))
    cl.user_session.set("history", history)
