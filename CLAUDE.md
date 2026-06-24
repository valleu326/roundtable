# 圆桌 RoundTable

> 本文件是项目纲领,给后续的 Claude 会话(以及你自己)快速理解这个项目用。先读这一篇,再读代码。

## 这个项目是什么

「圆桌」是一个**多智能体辩论(Multi-Agent Debate, MAD)**问答应用,灵感来自 Andrej Karpathy 的 LLM Council,但针对**国内用户**做了适配:不依赖 OpenRouter,直接用国内模型服务商。

一句话:**你问一个问题,多个 AI 模型围着"圆桌"各自作答、参考他人回答后修正自己,最后由一个"主席"模型打分排序、合并出一份更全面的最终答案给你。**

之所以叫"圆桌",取意大家都坐在桌前、平等辩论、博采众方。

## 技术栈与各自职责

| 库 | 作用 | 在项目里的角色 |
|---|---|---|
| **LiteLLM** | 用统一接口调用不同厂商的模型 | `llm.py` 封装一个 `call_llm()`,屏蔽各家 API 差异 |
| **LangGraph** | 把工作流画成"图"来编排 | `graph.py` 定义 4 个节点串成一条线,跑完一轮辩论 |
| **langchain-mcp-adapters** | 给模型挂 MCP 工具(搜索/代码/shell) | **第一版不用**,留到第二版 |
| **Chainlit** | 纯 Python 的聊天界面 | `app.py` 做前端:展示最终答案 + 实时推进 4 阶段进度 |

## 第一版边界(MVP)

第一版只做**纯文本的辩论闭环**,刻意控制范围,先把骨架跑通:

- ✅ 纯文本,不接 MCP 工具
- ✅ 修正形式:每个 agent 看**除自己外其他人(匿名)的初始回答**,参考后直接修正自己(不再做两两互评;原方案互评耗 N×(N-1) 次调用且只能看二手评价,现为 N 次调用、看第一手回答)
- ✅ 打分排序 + 合并最终答案:**单独一个"主席"模型**当裁判(可选;不配则默认复用第一个选手)
- ✅ **已支持多厂商混搭**:DeepSeek / 通义千问 / GLM / Kimi / 豆包 等,在 `config.yaml` 里各填各的连接信息即可。多样性既可靠"多厂商",也可靠"同一模型 + 不同角色人设/温度"
- ✅ API key 直接写在 `config.yaml` 的每条 `litellm_params` 里,不需要环境变量、不需要 `.env`

❌ 第一版不做:知识库 RAG、多轮辩论循环、MCP 工具、UI 可调节点数/温度的控件。

## 一轮辩论的 4 个阶段(就是 LangGraph 的 4 个节点)

```
用户问题
  ↓
① 各自作答    N 个 agent 并发各答一次
  ↓
② 各自修正    每个 agent 看到他人(匿名编号)的初始回答,参考后修正自己 —— N 次并发
  ↓
③ 主席打分    主席给每个修正回答打分,输出 JSON 并排序
  ↓
④ 主席合并    主席把所有修正回答合并成最终答案
  ↓
返回给用户: 最终答案 + 等待时实时推进的 4 阶段进度(哪个在跑/已完成/卡住)
```

> **为什么节点内部用 `asyncio.gather` 并发,而不把每个 agent 拆成 LangGraph 节点?**
> 为了让图只有 4 个节点、一眼能看懂。LangGraph 负责阶段编排,节点内部的"同时对 N 个模型发请求"交给 asyncio。注释里有说明,也可以改成 LangGraph 的并行边(`add_edge(["n1","n2"], "merge")`)。

## 会话上下文规则(重要)

- **只有"用户问 + 最终答案"这一对**会进入会话历史,喂给后续轮次作为上下文。
- 初始回答、修正等**中间过程不进**会话上下文,只在 Chainlit 界面的进度面板里给人看。
- 这样既能"博采众方",又不会把上下文撑爆。

## 配置怎么管:config.yaml + config.py

项目配置分两个文件,各司其职:

- **`config.yaml`**:只管"有哪些模型、各连哪家、key 是什么"——傻瓜式配置,照着加删即可。这是**唯一需要你动手填的文件**,而且因为含 key,**不进 git**(已在 `.gitignore`)。
- **`config.py`**:负责把 `config.yaml` 读进来,加上行为常量(超时 `MODEL_TIMEOUT`、重试 `MAX_RETRIES`、历史轮数 `MAX_HISTORY_TURNS`),并提供 `check_ready()` 启动自检。代码里 `import config` 时就会加载 yaml,找不到文件会直接报错。

`config.yaml` 的结构完全对标 LiteLLM 官方写法:

```yaml
model_list:                 # 参赛选手,条数 = 选手数
  - model_name: DeepSeek     # UI 显示名 / 内部 id
    persona: ""              # 人设,可留空
    temperature: 1.0
    litellm_params:
      model: deepseek-v4-pro  # 厂商模型名,见下方"模型名"说明
      api_base: https://api.deepseek.com
      api_key: sk-xxxx        # 真实 key,直接写在这

chair:                       # 可选:主席模型。不配则默认用第一个选手
  model_name: 主席
  litellm_params:
    model: deepseek-v4-pro
    api_base: https://api.deepseek.com
    api_key: sk-xxxx
```

仓库不自带 `config.yaml`,而是给了脱敏模板 **`config.example.yaml`**。新用户第一步:

```bash
cp config.example.yaml config.yaml   # 然后编辑 config.yaml 填 key
```

> **模型名 `model` 字段的规范**(`config.py` 的 `_normalize_model` 统一处理):
> 各厂商 OpenAI 兼容接口,用户在 yaml 里写的 model 名五花八门。统一做法是剥掉可能的 `openai/` 前缀后再统一加上 `openai/`,最终变成 `openai/<用户写的整串>`。这样:
> - 裸名 `deepseek-v4-pro` → `openai/deepseek-v4-pro`
> - 带厂商斜杠 `zai-org/GLM-5.2` → `openai/zai-org/GLM-5.2`(厂商斜杠原样保留)
> - 凡以 `openai/` 开头,litellm 就走 OpenAI 兼容路由,用本条的 `api_base` + `api_key` 去调用,模型名原样传给厂商。
> - 不做这步的话,litellm 会把第一段(如 `zai-org`)当成它自己的 provider 前缀,就不走我们的 api_base 了。

## 目录结构与各文件职责

```
RoundTable/
├── CLAUDE.md              ← 你正在读的这篇,项目纲领
├── README.md              ← 怎么装、怎么跑(面向使用者)
├── requirements.txt       ← 依赖与版本
├── config.example.yaml    ← 配置模板(脱敏示例,复制成 config.yaml 用)
├── config.yaml            ← 真实配置(含 key,不进 git,需自行创建)
├── run.sh                 ← 一键启动脚本(source 即用)
├── chainlit.md            ← Chainlit 首屏欢迎页内容
├── config.py              ← 配置:加载 config.yaml + 行为常量 + 启动自检(改常量动这里)
├── llm.py                 ← LiteLLM 封装:统一的 call_llm()
├── prompts.py             ← 4 阶段的中文提示词模板
├── state.py               ← LangGraph 的状态定义(TypedDict)
├── chair.py               ← "主席"裁判:打分 + 合并 + JSON 解析
├── graph.py               ← LangGraph 图:4 节点串成链,编译成 app
├── debate.py              ← 桥:把(历史+问题)喂给图,跑完返回完整结果
├── app.py                 ← Chainlit 入口:前端展示
└── .chainlit/             ← Chainlit 配置(config.toml;translations 已忽略)
```

阅读建议顺序(从简单到核心):
1. `config.py` → 2. `llm.py` → 3. `prompts.py` → 4. `state.py`
→ 5. `chair.py` → 6. `graph.py` → 7. `debate.py` → 8. `app.py`

## 怎么跑

```bash
conda activate roundtable          # 用已建好的 conda 环境
cp config.example.yaml config.yaml # 从模板创建配置(首次)
# 编辑 config.yaml,把每条 api_key 占位符换成真实 key
chainlit run app.py                # 启动,浏览器自动打开
# 或:source run.sh                 # 在已 activate 的终端里一键启动
```

直跑图做命令行自测(不走 UI):
```bash
python graph.py                    # 跑完整 4 阶段,终端打印全过程
```

## 开发约定

- **中文注释**为主,写给"不懂这三个库"的人看,关键处解释"为什么这么写"。
- **模块化、多文件**,每个文件职责单一、可单独读懂。
- **简约不简单**:核心功能做扎实,次要的先不写;不追求完美,迭代式前进。
- 加减选手 / 换模型 / 改人设 / 改温度 / 指定主席 —— **只动 `config.yaml`**,不动其它文件。
- 改超时/重试/历史轮数等行为常量 —— 动 `config.py` 的常量区。
- 改提示词文风、评判规则 —— 动 `prompts.py` / `chair.py`。

## 已实测的环境与 API(写代码时对齐过真实签名,非猜)

- conda 环境 `roundtable`,Python 3.12.13
- litellm 1.89.3 / langgraph 1.2.6 / langchain-core 1.4.8 / chainlit 2.11.1 / langchain-mcp-adapters 0.3.0 / pyyaml 6.0.3
- 已知避坑点(代码注释里会再提):
  - litellm `acompletion` 用 `base_url=`(不是老文档的 `api_base=`)、`api_key=`,首参是 `model`。
  - DeepSeek OpenAI 兼容接口:`api_base="https://api.deepseek.com"`,模型名如 `deepseek-v4-pro`。
  - **多厂商走 litellm 的 OpenAI 兼容路由**:yaml 里 `model` 写厂商模型名(裸名或带斜杠均可),`config.py` 的 `_normalize_model` 自动加 `openai/` 前缀,再用该条的 `api_base` + `api_key` 调用。这样同一批模型都来自硅基流动也没关系,各记自己的连接信息。详见上文「配置怎么管」。
  - Chainlit `cl.Step(type="run"|"tool"|"llm", default_open=False, auto_collapse=False)`,支持嵌套;`cl.user_session` 存跨轮历史;首屏欢迎页内容来自根目录 `chainlit.md`。
  - LangGraph `StateGraph(state_schema=State)` + `add_node(func)`(函数名即节点名)+ `add_edge` + `compile()`。

## 未来计划(尚未实现,别误以为是已有功能)

- [ ] 接 MCP 工具:网页搜索 / 写代码 / 执行 shell(chainlit 2.x 自带 `cl.mcp`,可省一层 adapter)
- [ ] 知识库 RAG 检索
- [ ] 评分策略细化、多轮辩论循环、UI 上可调节点数/温度的控件

> 注:**多厂商混搭已在第一版落地**(改 `config.yaml` 即可,图不动),所以不再列入未来计划。
