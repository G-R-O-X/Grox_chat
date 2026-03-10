# Agent Chatroom (多智能体科研推理系统)

[English](README.md)

一个以数据库为中心、由图状态机调度的多代理推理系统，面向长时间、多阶段的技术讨论与知识沉淀。

### 系统简介

Agent Chatroom 不是普通的多人闲聊，而是一个围绕 SQLite 持久化黑板运行的结构化推理竞技场。

- `观众 (Audience)` 负责总体规划、为每个子题撰写背景简报、总结进度，并决定何时终止。
- 专家组负责推动核心推理：`空想家 (Dreamer)`、`科学家 (Scientist)`、`工程师 (Engineer)`、`分析师 (Analyst)`、`批评家 (Critic)` 和 `少数派 (Contrarian)`。
- `猫 (Cat)`、`狗 (Dog)` 和 `裁决者 (Tron)` 组成异步的验证与反馈层。
- `作家 (Writer)` 负责核查硬性断言，并将稳定的结论转化为可复用的事实。
- 检索是显式的：代理会先决定查询什么，然后执行向量化、搜索、重排，最后才开始发言。

### 执行模型

- 每个发言角色在生成消息前都必须先走一遍本地 RAG。
- `第 1 轮 / opening`：`空想家 -> 科学家 -> 工程师 -> 分析师 -> 批评家 -> 裁决者`
- `第 2 轮 / evidence`：`空想家 -> 科学家 -> 工程师 -> 分析师 -> 批评家 -> 少数派 -> 狗 -> 猫 -> 裁决者`
- `第 3 轮起 / debate`：基础 roster 与第二轮相同，并在轮末按固定顺序追加额外回合 `裁决者 -> 狗 -> 猫`
- 外部 web search 按 phase 控制：第一轮关闭，第二轮全员可选，第三轮起只对特定角色开放
- `作家 (Writer)` 每轮末都会执行，并可把核验后的事实写入 Fact 表

### 系统架构

```mermaid
flowchart LR
    User[用户 / 输入话题] --> Audience[观众<br/>规划, 背景简报,<br/>总结, 终止判定]
    Audience --> Arena[子题讨论区]
    Arena --> Writer[作家<br/>事实核查]
    Writer --> Audience

    Memory[(SQLite 黑板<br/>Topic / Plan / Subtopic / Message / Fact<br/>vec_facts / vec_messages)]

    Audience <--> Memory
    Arena <--> Memory
    Writer <--> Memory
```

### 子题循环工作流

```mermaid
flowchart TD
    Grounding[观众生成背景简报] --> Opening[第 1 轮 / opening<br/>空想家 -> 科学家 -> 工程师 -> 分析师 -> 批评家 -> 裁决者<br/>仅本地 RAG]
    Opening --> Evidence[第 2 轮 / evidence<br/>空想家 -> 科学家 -> 工程师 -> 分析师 -> 批评家 -> 少数派 -> 狗 -> 猫 -> 裁决者<br/>本地 RAG + 可选外搜]
    Evidence --> Debate[第 3 轮起 / debate<br/>基础 roster 保持不变<br/>本地 RAG 始终开启]
    Debate --> Extra[额外发言回合<br/>裁决者修复 -> 狗纠错 -> 猫扩展]
    Extra --> Writer[作家事实核查<br/>每轮执行]
    Writer --> Summary[观众生成总结]
    Summary --> Eval[观众终止判定<br/>结合历史总结库RAG]
    Eval -->|继续| Next[准备下一轮]
    Next --> PhaseShift{进入下一阶段}
    PhaseShift -->|第 2 轮| Evidence
    PhaseShift -->|第 3 轮起| Debate
    Eval -->|结束| End[关闭子题]
```

### 角色分配

- `观众 (Audience)`：主持人、规划者、总结者和话题层级的总控。
- `作家 (Writer)`：事实核查员与知识提炼者；负责向事实库注入数据。
- `空想家 (Dreamer)`：提出新方向和大胆假设。
- `科学家 (Scientist)`：检验机制、理论和内部有效性。
- `工程师 (Engineer)`：把想法转成可实现系统和执行路径。
- `分析师 (Analyst)`：提供指标、概率、不确定性和数据视角。
- `批评家 (Critic)`：强力攻击漏洞、边界条件和逻辑断裂。
- `少数派 (Contrarian)`：主动反对正在形成的主流共识。
- `猫 (Cat)`：挑出最有价值的发言，并给予额外回合（奖励）。
- `狗 (Dog)`：挑出最可疑的发言，并强制其补充自证（惩罚）。
- `裁决者 (Tron)`：执行论坛法则，处理严重幻觉、偏置和逻辑失控。

### 检索与记忆

系统维护两条长期记忆通道：

- `Fact RAG`：由 `Writer` 提炼出的可复用硬事实。
- `Summary RAG`：历史总结记忆，用于检测重复、停滞和语义回环。

理想检索路径会在每次发言前执行：

1. 先根据当前角色和争论生成检索问题。
2. 对问题做 embedding 向量化。
3. 从本地向量库召回候选记忆。
4. 用 reranker 交叉编码器精排。
5. 把最相关证据注入下一次发言的 prompt。

### 项目结构

- `src/agent_chatroom/`: 调度、模型客户端、检索、持久化、prompt
- `tests/`: 单元测试与集成测试
- `DESIGN.md`: 完整设计文档

### 快速开始

```bash
uv sync
cp .env.example .env
uv run python -c "from agent_chatroom.db import init_db; init_db()"
uv run python -m agent_chatroom.server
```

另开一个终端创建 topic：

```bash
uv run python -c "from agent_chatroom.api import create_topic; create_topic('主题摘要', '更详细的主题描述')"
```

### 快速冒烟测试

你可以用一个故意荒谬但中性的题目做快速检查：

```bash
uv run python -c "from agent_chatroom.api import create_topic; create_topic('从职场实践角度出发，早上进门应该先左脚还是先右脚？', '从职场实践角度出发，早上进门应该先左脚还是先右脚？')"
```

另开一个终端查看数据库状态：

```bash
uv run python -c "import sqlite3; conn = sqlite3.connect('chatroom.db'); print(conn.execute('select count(*) from Subtopic').fetchone()[0], conn.execute('select count(*) from Message').fetchone()[0], conn.execute('select count(*) from Fact').fetchone()[0])"
```

运行测试：

```bash
uv run pytest -q
```
