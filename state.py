"""
state.py —— LangGraph 的“状态”定义。

什么是状态？可以把 LangGraph 想象成一条流水线，每个节点（工位）看完上一步的半成品、
添上自己这一步的成果，再传给下一步。这个“在工位之间传递的半成品”就是状态(State)。

我们用 Python 的 TypedDict 来定义它：相当于声明这个状态对象里有哪些字段、各是什么类型。
4 个节点依次往里填字段：
  ① initial 节点填 initial        —— 各 agent 的初始回答
  ② revise 节点填 revised         —— 各 agent 修正后的回答
  ③ score 节点填 scores           —— 主席打的分(已排序)
  ④ merge 节点填 final_answer     —— 最终合并答案

关于 reducer（合并函数）：
LangGraph 里，如果某个字段会被多个并行节点同时写，需要告诉它“冲突时怎么合并”。
第一版我们的节点是串行的(一个接一个)，不会并发写同一字段，所以 reducer 用最简单的
“后者覆盖前者”即可——这里只为符合 LangGraph 的规范留好扩展位。
"""

from typing import Annotated, TypedDict


def _overwrite(left, right):
    """
    最简单的合并策略：直接用新值覆盖旧值。
    （LangGraph 默认就是这样，但显式写出来更清晰，将来要改成“合并”也只动这里。）
    """
    return right if right is not None else left


class DebateState(TypedDict, total=False):
    """
    圆桌辩论的全部状态字段。total=False 表示每个字段都不是必填，
    节点按需更新自己负责的那几个字段即可。

    Annotated[类型, _overwrite] 的意思是：
    这个字段用 _overwrite 策略合并——第一版就是覆盖，留作以后扩展。
    """
    # 用户这一轮的问题（整轮不变）
    question: Annotated[str, _overwrite]

    # 历史对话上下文字符串（由 app.py 拼好传进来，整轮不变）
    history_str: Annotated[str, _overwrite]

    # ① 各 agent 的初始回答：{agent_id: 回答文本}
    initial: Annotated[dict, _overwrite]

    # ② 各 agent 修正后的回答：{agent_id: 修正回答文本}
    revised: Annotated[dict, _overwrite]

    # ③ 主席打分结果：[{"id","score","reason"}, ...] 已按分数降序排好
    scores: Annotated[list, _overwrite]

    # ④ 最终合并答案（返回给用户的）
    final_answer: Annotated[str, _overwrite]
