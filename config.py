"""
config.py —— 项目的常量与配置加载入口。

分工（和 models.yaml 分开，方便阅读）：
  - models.yaml  : 只管“有哪些模型、各连哪家、key 是什么”——傻瓜式配置，照着加删即可。
  - config.py    : 负责把 models.yaml 读进来，加上一些行为常量（超时、重试、历史轮数等），
                   并提供方便的取用函数。

设计要点：
  - 选手数量 = models.yaml 里 model_list 的条数，配几个就是几个，不写死。
  - 每个选手自带 api_base + api_key + model，不依赖任何“厂商名”或全局 key。
  - 这样几个模型都来自硅基流动也没关系，它们各记自己的连接信息即可。
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


# ──────────────────────────────────────────────
# 1. 加载 models.yaml
# ──────────────────────────────────────────────
# yaml 文件就放在本文件同级目录。读一次，缓存起来用。
_CONFIG_PATH = Path(__file__).parent / "models.yaml"


def _load_yaml() -> dict:
    """读取 models.yaml，返回原始字典。读不到就给个清晰报错。"""
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"没找到配置文件 {_CONFIG_PATH}。请先创建 models.yaml（可参考 README）。"
        )
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_RAW = _load_yaml()


# ──────────────────────────────────────────────
# 2. 选手模型（参与作答 + 参考他人修正）
# ──────────────────────────────────────────────

@dataclass
class AgentConfig:
    """
    一个参赛选手。字段全部直接来自 models.yaml 的一条 model_list 记录。

    - name       ：UI 显示名 / 内部 id（取自 model_name）。
    - persona    ：中文人设，写进提示词，让不同模型有不同风格。
    - temperature：温度。越高越发散，越低越稳定。
    - model      ：litellm 实际用的“模型名”（已带 openai/ 前缀）。
    - api_base   ：该模型的 API 地址（OpenAI 兼容）。
    - api_key    ：该模型的 API 密钥（直接写在 yaml 里）。
    """
    name: str
    persona: str
    temperature: float
    model: str        # 形如 openai/deepseek-v4-pro
    api_base: str
    api_key: str


def _normalize_model(raw_model: str) -> str:
    """
    把 yaml 里写的模型名，统一规范成 litellm 需要的格式：「openai/<原始模型名>」。

    为什么要做这一步？
    各厂商的 OpenAI 兼容接口，用户在 yaml 里写的 model 名五花八门：
      - 裸名：        deepseek-v4-pro
      - 带厂商斜杠：  zai-org/GLM-5.2、moonshotai/Kimi-K2.7-Code
      - 可能误带：    openai/xxx
    litellm 的规则：以 "openai/" 开头 = 走 OpenAI 兼容路由（这时它会用我们给的
    api_base + api_key 去调，模型名原样传给厂商）。否则 litellm 会把第一段当
    成它自己的 provider 前缀（如把 "zai-org" 当 provider），就不走我们的 api_base 了。

    所以统一做法：剥掉用户可能写的 "openai/" 前缀，再统一加上 "openai/"，
    最终变成 openai/<用户写的整串>。这样厂商斜杠（如 zai-org/GLM-5.2）被原样保留。
    """
    m = str(raw_model or "").strip()
    if m.startswith("openai/"):
        m = m[len("openai/"):]
    return f"openai/{m}" if m else ""


def _build_agents(raw_list: list) -> list[AgentConfig]:
    """把 yaml 的 model_list 逐条转成 AgentConfig。自动适应条数。"""
    agents: list[AgentConfig] = []
    for idx, item in enumerate(raw_list):
        params = item.get("litellm_params", {}) or {}
        agents.append(
            AgentConfig(
                name=str(item.get("model_name") or f"agent{idx}"),
                persona=str(item.get("persona") or ""),
                temperature=float(item.get("temperature", 0.7)),
                model=_normalize_model(params.get("model", "")),
                api_base=str(params.get("api_base", "")),
                api_key=str(params.get("api_key", "")),
            )
        )
    return agents


# 所有参赛选手。配几个就是几个，图/UI 会自动适应。
AGENTS: list[AgentConfig] = _build_agents(_RAW.get("model_list", []) or [])


# ──────────────────────────────────────────────
# 3. “主席”裁判模型（负责打分排序 + 合并最终答案）
# ──────────────────────────────────────────────
# 主席单独在 models.yaml 的 chair 节里配（结构和选手一样），用稳定低温调用。
# 不配的话，默认复用第一个选手当主席，省得最小配置也能跑。

@dataclass
class ChairConfig:
    name: str
    model: str
    api_base: str
    api_key: str


def _build_chair() -> ChairConfig:
    chair_raw = _RAW.get("chair") or {}
    if chair_raw:  # yaml 里显式配了主席
        params = chair_raw.get("litellm_params", {}) or {}
    elif AGENTS:   # 没配就默认用第一个选手
        a = AGENTS[0]
        return ChairConfig(name=a.name, model=a.model, api_base=a.api_base, api_key=a.api_key)
    else:
        raise ValueError("models.yaml 既没有 model_list 也没有 chair，没法跑。")
    return ChairConfig(
        name=str(chair_raw.get("model_name") or "主席"),
        model=_normalize_model(params.get("model", "")),
        api_base=str(params.get("api_base", "")),
        api_key=str(params.get("api_key", "")),
    )


CHAIR: ChairConfig = _build_chair()
CHAIR_TEMPERATURE = 0.0  # 主席要稳定客观，温度设 0


# ──────────────────────────────────────────────
# 4. 行为常量
# ──────────────────────────────────────────────

MODEL_TIMEOUT = 60   # 调用超时（秒）
MAX_RETRIES = 2      # 失败重试次数
MAX_HISTORY_TURNS = 6  # 保留多少轮历史问答作为上下文


# ──────────────────────────────────────────────
# 5. 启动前的就绪自检
# ──────────────────────────────────────────────

def check_ready() -> list[str]:
    """
    检查配置是否够开辩：至少 2 个选手（才能各自看到他人回答做参考修正）、每个都有 model/api_base/api_key。
    返回缺失说明列表：空 = 就绪；非空 = 每条说明缺什么。app.py 启动时会提示给用户。
    """
    missing = []
    if len(AGENTS) < 2:
        missing.append(f"选手太少：目前只有 {len(AGENTS)} 个，至少需要 2 个才能参考互修正")
    for a in AGENTS:
        if not a.model:
            missing.append(f"选手「{a.name}」缺少 model")
        if not a.api_base:
            missing.append(f"选手「{a.name}」缺少 api_base")
        if not a.api_key or "在此" in a.api_key:
            missing.append(f"选手「{a.name}」的 api_key 还没填（yaml 里仍是占位符）")
    if not CHAIR.model or not CHAIR.api_key:
        missing.append("主席模型配置不完整")
    return missing
