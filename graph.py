"""
graph.py —— 用 LangGraph 把“圆桌辩论”画成一张图。

不懂 LangGraph 的话，先这样理解：
- 一个“图”= 一条流水线，由若干“节点(工位)”用“边(箭头)”连起来。
- 状态(DebateState)就是流水线上传送的半成品，每过一个节点就被加工一次。
- 我们有 4 个节点，顺序连成一条线：

  START → ① 各自作答 → ② 各自修正(参考他人回答) → ③ 主席打分 → ④ 主席合并 → END

  说明：第一版曾设计「两两互评」环节(节点②)，但那一步要 N×(N-1) 次调用、且每个模型只能
  看到别人对自己的【评价】(二手信息)。现在改为：作答后直接把【其他人的原始回答】喂给每个
  模型，让它参考第一手资料修正自己——只需 N 次调用，信息更全更准，图也更简洁。

- 节点里“同时调用多个模型”的部分用 asyncio.gather 并发，
  没有把每个 agent 拆成单独的图节点——这样图只有 4 个节点，一眼能看懂。
  （将来想用 LangGraph 原生并行边，可以改成 add_edge(["n1","n2"], "merge")，逻辑等价。）

每个节点函数的写法约定：
  async def node_xxx(state, config) -> dict:
      ...
      return {"某字段": 值}   # 只返回本节点要更新的字段
LangGraph 会把返回的 dict 合并进状态，传给下一个节点。
"""

import asyncio

from langgraph.graph import START, END, StateGraph

import config
import llm
import prompts
from chair import chair_merge, chair_score
from state import DebateState


# ──────────────────────────────────────────────
# ① 各自作答
# ──────────────────────────────────────────────

async def node_initial(state: DebateState, _config=None) -> dict:
    """
    每个 agent 独立回答用户问题，并发执行。
    返回 {initial: {agent_id: 回答}}。
    """
    question = state["question"]

    async def one(agent: config.AgentConfig) -> tuple[str, str]:
        answer = await llm.call_llm(
            system=prompts.initial_system(agent.persona, []),
            user=prompts.initial_user(question),
            temperature=agent.temperature,
            model=agent.model, api_base=agent.api_base, api_key=agent.api_key,
        )
        return agent.name, answer

    # 并发跑所有 agent 的作答。gather 会等它们全部返回。
    results = await asyncio.gather(*[one(a) for a in config.AGENTS])
    return {"initial": dict(results)}


# ──────────────────────────────────────────────
# ② 各自修正（参考他人回答后直接修正）
# ──────────────────────────────────────────────

async def node_revise(state: DebateState, _config=None) -> dict:
    """
    每个 agent 看到【除自己外其他人】的初始回答(匿名编号)，参考后修正自己的回答。
    返回 {revised: {name: 修正回答}}。

    替代的「两两互评」环节：原来每个 agent 只能看到别人对自己的【评价】(二手信息)，
    还要 N×(N-1) 次调用。现在直接把别人的【原始回答】喂给它，第一手信息更全更准，
    且只需 N 次并发调用。

    匿名：给每个 agent 的“他人回答列表”按 config.AGENTS 的固定顺序、排除掉自己后，
    局部重新编号为 选手1/选手2…。这样不同 agent 看到的“选手1”指的是不同的人，
    不会暴露他人姓名/人设，保持中立。
    """
    question = state["question"]
    initial = state["initial"]  # {name: 初始回答}

    async def one(agent: config.AgentConfig):
        # 其他人：固定顺序、排除自己。编号是相对于「当前这个 agent」的，所以匿名。
        others = [
            (f"选手{i+1}", initial[other.name])
            for i, other in enumerate(config.AGENTS)
            if other.name != agent.name
        ]
        revised = await llm.call_llm(
            system=prompts.revise_system(agent.persona, []),
            user=prompts.revise_user(
                question,
                my_answer=initial[agent.name],
                others=others,
            ),
            temperature=agent.temperature,
            model=agent.model, api_base=agent.api_base, api_key=agent.api_key,
        )
        return agent.name, revised

    results = await asyncio.gather(*[one(a) for a in config.AGENTS])
    return {"revised": dict(results)}


# ──────────────────────────────────────────────
# ③ 主席打分
# ──────────────────────────────────────────────

async def node_score(state: DebateState, _config=None) -> dict:
    """主席给修正回答打分排序。返回 {scores: [{id,score,reason}, ...]}。"""
    scores = await chair_score(state["question"], state["revised"])
    return {"scores": scores}


# ──────────────────────────────────────────────
# ④ 主席合并
# ──────────────────────────────────────────────

async def node_merge(state: DebateState, _config=None) -> dict:
    """主席把排序后的修正回答合并成最终答案。返回 {final_answer: 文本}。"""
    final = await chair_merge(state["question"], state["scores"], state["revised"])
    return {"final_answer": final}


# ──────────────────────────────────────────────
# 把 4 个节点连成图
# ──────────────────────────────────────────────

def build_graph():
    """
    构建并编译 LangGraph 图，返回可调用的 app。

    add_node(节点函数)：注册节点，节点名 = 函数名(所以要保证函数名唯一)。
    add_edge(START, "node_initial")：图从 node_initial 开始。
    后面依次用 add_edge 把 4 个节点串成一条线。
    compile()：编译成可执行的 Runnable，之后用 app.ainvoke(初始state) 跑。
    """
    builder = StateGraph(DebateState)
    builder.add_node(node_initial)
    builder.add_node(node_revise)
    builder.add_node(node_score)
    builder.add_node(node_merge)

    builder.add_edge(START, "node_initial")
    builder.add_edge("node_initial", "node_revise")
    builder.add_edge("node_revise", "node_score")
    builder.add_edge("node_score", "node_merge")
    builder.add_edge("node_merge", END)

    return builder.compile()


# 模块加载时就编译好，供 debate.py 直接 import 使用。
app = build_graph()


# ──────────────────────────────────────────────
# 命令行自测：直接 `python graph.py` 可跑通整条 4 阶段链路(不走 UI)。
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    import config

    async def _smoke():
        missing = config.check_ready()
        if missing:
            print("⚠️  配置未就绪：")
            for m in missing:
                print("   -", m)
            return

        question = "用一句话解释什么是多智能体辩论，并说说它有什么用。"
        print(f"→ 开始一轮圆桌辩论，问题：{question}\n")

        # 直接复用 debate 的编排逻辑跑一遍。
        from debate import run_debate
        result = await run_debate(question, history=[])

        print("=" * 60)
        print("① 各 agent 初始回答：")
        for aid, ans in result.get("initial", {}).items():
            print(f"  [{aid}] {ans[:80]}...")
        print("\n② 各 agent 修正回答(参考他人后)：")
        for aid, ans in result.get("revised", {}).items():
            print(f"  [{aid}] {ans[:80]}...")
        print("\n③ 主席打分(排序后)：")
        for s in result.get("scores", []):
            print(f"  [{s.get('name', s.get('slot',''))}] {s['score']} 分 —— {s.get('reason','')[:50]}")
        print("\n④ 最终合并答案：")
        print(result.get("final_answer", "(空)"))

    asyncio.run(_smoke())

