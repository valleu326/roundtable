"""
llm.py —— 对 LiteLLM 的一层薄封装。

全项目只有一个函数 call_llm()，所有“调用大模型”的地方都走它。
好处：① 屏蔽各家 API 差异；② 统一加超时/重试；③ 想记录日志、换底层都只改这里。

不懂 LiteLLM 的话记住一点：它是个“统一插座”，不管 DeepSeek 还是千问，
都用同一个 acompletion() 函数，靠 base_url + api_key 区分去哪家电。
"""

import asyncio
from typing import Optional

import litellm

import config


# 关掉 litellm 那些琐碎的打印，让控制台干净点。
litellm.suppress_debug_info = True


async def call_llm(
    system: str,
    user: str,
    *,
    temperature: float,
    model: str,
    api_base: str,
    api_key: str,
    max_tokens: Optional[int] = None,
) -> str:
    """
    调用一次大模型，返回纯文本回答。

    参数：
        system：系统提示词（设定角色/规则）。
        user：用户输入这一轮的具体内容。
        temperature：温度。越高越发散，越低越稳定。
        model：litellm 模型名（已带 openai/ 前缀，如 "openai/deepseek-v4-pro"）。
        api_base：该模型的 OpenAI 兼容 API 地址。
        api_key：该模型的 API 密钥。
        max_tokens：回答最大长度上限，None 表示不限。

    返回：模型的纯文本回答字符串。

    注意：litellm Python SDK 的参数叫 base_url（不是 yaml 里的 api_base），
          传入前要改个名——这是 litellm 自己两套命名的小坑。
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # 失败就重试 MAX_RETRIES 次，每次退避等待，避免连环撞限流。
    last_err: Exception | None = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=config.MODEL_TIMEOUT,
                base_url=api_base,
                api_key=api_key,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as err:  # noqa: BLE001 —— 调用失败原因多样，统一捕获重试
            last_err = err
            if attempt < config.MAX_RETRIES:
                await asyncio.sleep(1.0 * (attempt + 1))

    raise RuntimeError(f"调用模型 {model} 失败：{last_err}")


# ──────────────────────────────────────────────
# 自测：直接 `python llm.py` 验证第一个选手能否调通。
# ──────────────────────────────────────────────
if __name__ == "__main__":
    async def _smoke():
        missing = config.check_ready()
        if missing:
            print("⚠️  配置未就绪：")
            for m in missing:
                print("   -", m)
            return
        a = config.AGENTS[0]
        print(f"→ 用选手「{a.name}」({a.model}) 调用测试中……")
        answer = await call_llm(
            system="你是个简洁的助手，回答不超过30个字。",
            user="用一句话解释什么是多智能体辩论。",
            temperature=0.3,
            model=a.model, api_base=a.api_base, api_key=a.api_key,
        )
        print("✅ 调用成功，回答：")
        print(answer)

    asyncio.run(_smoke())
