# 圆桌 RoundTable　`v0.1`

> 多智能体辩论(MAD)问答应用：你问一个问题，多个 AI 模型围着「圆桌」各自作答、参考他人的回答后修正自己，最后由「主席」打分排序、合并出一份更全面的最终答案。仿 Andrej Karpathy 的 LLM Council，但针对国内用户，直接用国内模型服务商。

第一版是**纯文本辩论闭环**，已跑通。**已支持多厂商混搭**（DeepSeek / 通义千问 / GLM / Kimi / 豆包，照着配置填即可）。后续会迭代加入 MCP 工具、RAG 等（见 CLAUDE.md「未来计划」）。

---

## 工作流（一轮辩论的 4 个阶段）

```
用户问题
  ↓
① 各自作答    N 个 agent 并发各答一次
  ↓
② 各自修正    每个 agent 看到「除自己外其他人(匿名)」的初始回答，参考后修正自己 —— N 次并发
  ↓
③ 主席打分    主席给每个修正回答打分，输出 JSON 并排序
  ↓
④ 主席合并    主席把所有修正回答合并成最终答案
  ↓
返回： 最终答案(主对话流) + 等待时实时显示的 4 阶段进度
```

> 第一版曾在 ①② 之间设计过「两两互评」环节，但那一步要 N×(N-1) 次调用、且模型只能看到别人对自己的二手评价；现在直接把同行的原始回答喂给彼此参考，调用更少、信息更全。

> 等待期间界面会实时反馈 LangGraph 跑到哪一步了（① 作答 → ② 修正 → ③ 评分 → ④ 合并，哪步亮哪步），不会全程黑盒。

**会话上下文规则**：只有「问 + 最终答案」进历史，喂给后续轮次；中间过程只在进度面板里展示，不进上下文。

---

## 快速开始

### 1. 准备环境（用 conda）

已装 MiniConda 的话：

```bash
conda create -n roundtable python=3.12 -y
conda activate roundtable
pip install -r requirements.txt
```

> 没有 conda 也能用 venv：`python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

### 2. 配置模型（创建你的 models.yaml）

仓库不会自带 `models.yaml`（因为它要放你的 API key，不提交）。改用模板生成：

```bash
cp models.example.yaml models.yaml
```

然后打开 [models.yaml](models.yaml)，把每一条的 `api_key:` 占位符换成你的真实 key。模板里默认启用了 2 个选手（DeepSeek、GLM），还注释了通义千问 / Kimi / 豆包的写法，想用谁就取消注释、填上 key。

每条选手需要填的东西：

```yaml
model_list:
  - model_name: DeepSeek            # 你起的名字，UI 上显示用
    persona: ""                     # 人设，可留空，也可写一句话让模型有不同风格
    temperature: 0.3                # 温度，越高越发散、越低越稳定
    litellm_params:
      model: deepseek-v4-pro        # 模型名（裸名或带厂商斜杠都行，会自动加 openai/ 走兼容协议）
      api_base: https://api.deepseek.com
      api_key: sk-你的真实key        # 直接粘贴，无需环境变量
```

> 加减选手 = 增删 `model_list` 条数，**选手数量 = 列表条数，随便几个都行**。
> 写法完全对标 LiteLLM 官方的 config.yaml 格式，将来也能直接喂给 litellm 本身。
> 主席模型（打分+合并）可选，不配 `chair:` 节就默认复用第一个选手。
> 几个模型都来自硅基流动也没关系——每个选手自带 api_base + api_key，按真实来源各填各的即可。

### 3. 启动

```bash
conda activate roundtable
chainlit run app.py
# 或者：source run.sh   （在已 activate roundtable 的终端里，自动切到项目目录启动）
```

浏览器自动打开（默认 http://localhost:8000）。直接提问即可，每轮约 30~90 秒。
**若 models.yaml 有没填的占位符，界面会明确提示缺哪个**，照提示补好重启即可。

### 4. 命令行自测（不走 UI，看完整过程）

```bash
python graph.py     # 跑完整 4 阶段，终端打印全过程
```

(若只想测某单个选手能不能调通，可在 Python 里 `import llm` 后用 `llm.call_llm(...)` 试一次。)

---

## 目录结构

```
RoundTable/
├── CLAUDE.md              项目纲领（先读这个）
├── README.md              你正在读的
├── requirements.txt       依赖
├── models.example.yaml    ★ 配置模板（脱敏示例，照着复制出 models.yaml）
├── models.yaml            ★ 你的真实配置（含 key，不会进 git，需自行创建）
├── run.sh                 一键启动脚本（source 即用）
├── config.py              加载 models.yaml + 行为常量（超时/重试/历史轮数）+ 启动自检
├── llm.py                 LiteLLM 封装：统一的 call_llm()
├── prompts.py             4 阶段中文提示词模板
├── state.py               LangGraph 状态定义
├── chair.py               主席裁判：打分 + 合并 + JSON 解析
├── graph.py               LangGraph 图：4 节点链 + 自测入口
├── debate.py              编排桥：跑图、返回完整结果
├── app.py                 Chainlit 前端入口
├── chainlit.md            Chainlit 首屏欢迎页内容
└── .chainlit/             Chainlit 配置（config.toml）
```

建议阅读顺序（从简到核心）：`config.py` → `llm.py` → `prompts.py` → `state.py` → `chair.py` → `graph.py` → `debate.py` → `app.py`

---

## 常见改法

| 想做的事 | 改哪里 |
|---|---|
| 加/减一个参赛选手 | `models.yaml` 的 `model_list`（复制一条 / 删一条） |
| 改某个选手的模型/地址/key | `models.yaml` 对应那条的 `litellm_params` |
| 改人设 / 温度 | `models.yaml` 对应那条的 `persona` / `temperature` |
| 单独指定主席模型 | `models.yaml` 末尾加一个 `chair:` 节（结构同选手） |
| 改提示词文风 / 评判规则 | `prompts.py` |
| 改界面展示 | `app.py` |
| 改超时/重试/历史轮数 | `config.py` 的常量区 |

---

## 已知的小提醒

- **没填 key / 占位符还在**：界面会明确提示缺哪个选手的 key，补好重启即可。
- **首轮较慢**：N 个选手约 N + N + 2 次模型调用（作答+参考修正+主席打分+合并），5 选手约 12 次，请耐心等。
- **API key 安全**：key 写在 `models.yaml` 里方便是方便，但该文件已在 `.gitignore` 中、不会进 git。请不要把它提交到公开仓库，也不要把真实 key 截图分享。
- **豆包模型名**：要用「推理接入点 endpoint id」(形如 ep-xxxx)，不是模型名，在火山方舟控制台创建后复制。
- **推理模型温度**：若接的是 deepseek-r1 等推理模型，temperature 可能被拒；第一版默认配的是普通对话模型，不受影响。

## 依赖版本（已实测）

- litellm 1.89.3 / langgraph 1.2.6 / langchain-core 1.4.8 / chainlit 2.11.1 / langchain-mcp-adapters 0.3.0 / pyyaml 6.0.3
- Python 3.12
