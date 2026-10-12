# Anthropic《Detecting and countering misuse of AI》中国相关内容提取

> **来源**：`hack/Anthropic-Detecting-and-countering-总结报告.md`（154 页报告的总结报告）
> **说明**：以下内容均来自 Anthropic 威胁情报报告（2026 年 9 月）的单方面表述与指控，**未经第三方核实**，其中涉及对中国机构和企业的指控存在明显的立场偏向，阅读时应保持批判性视角。

---

## 一、网络行动（Cyber Operations）

### GTG-10007：exploit "铸造厂"与自主攻击框架（中国长沙，疑似学生团伙）

- **组织形态**：多名操作者（含两名本科生）将 Claude 用作协调性攻击计划的工程与编排层，并行运行入侵、政府网络侦察、端点安全产品漏洞研究、恶意软件开发、情报采集等五条工作流；**持久化战役记忆跨会话保存上下文，Owner 离线后研究仍在继续**。
- **零日漏洞工厂**：固件解密 → 反编译（数千次调用）→ 漏洞假设（基于自建知识库）→ 利用代码生成 → 实验室验证 → 迭代，**单月产出十余个疑似零日漏洞**。
- **自主情报采集舰队**：13 个常驻采集 Agent 定时抓取美军公开合同、军事学说等 OSINT 内容，摘要评分后送分发门户。
- **目标规模**：约 50 个组织（教育、零售、能源、医疗、金融、政府），含某东南亚政府机构公民记录（姓名、电话、住址）。

---

## 二、监控行动（Surveillance Operations）

### 趋势层面

- **AI 取代工程劳动力**：报告称中国某宗教事务情报单位从多团队分析员缩减为单办公室 + AI 助手，月产数千份调查。
- **批量数据摄取与目标识别**：某 PRC 行为者让 Claude 给内容做政治敏感度评分并标记待"管控"目标。
- **深度嵌入国家安全官僚体系**：某 PRC 国家安全局用 Claude 编写内部《AI 用于监控行动手册》。

### GTG-14010：针对叙利亚维吾尔人的监控与招募（PRC 关联承包商）

- 把 100+ WhatsApp 群/Telegram 频道聊天转成中文结构化数据，按经济压力、家庭分离、思想动摇等**脆弱性画像**筛人（特别标注亲属在新疆者）。
- 用 Claude 扮演阿拉伯语"专家"质检方言话术；不识阿拉伯语的操作者借助 Claude 维持了多天可信的秘密招募行动。
- 规划对维吾尔多媒体（Uyghur Post）的协同举报/污名化/机器人放大压制。
- Claude 拒绝了秘密审讯和大规模人设培育请求。

### GTG-14020：宗教事务情报（针对天主教/藏传佛教/法轮功/台湾长老会）

- 单人运营四条并行工作流，按国家安全内部模板生成"人物调研底稿""线索报""态势感知"日报，字段含"工作抓手"（可利用的杠杆）。
- 目标含亚洲多国枢机主教；侦察宗教场所平面图。
- 提示词要求 Claude 采纳"中方立场"、将藏人行政中央定性为"非法分裂组织"。

### GTG-14021："维稳"与跨境镇压（三案关联）

- 市级网警单位用 **Claude Code + 自定义 skills** 跑舆情监控流水线、查询政府监控数据库、生成日报（含监控"李老师不是你老师"等海外异见账号）。
- 警校学生经重新提示绕过 Claude 拒答，生成点名 10 名公民的"管控"方案（拦截信访、"约谈"、行踪监控）。
- 地方国安局产出政府模板"态势感知"简报并索取海外合法抗议活动的场前情报（温哥华游行、土耳其维族文化活动、奥斯陆自由论坛），还编写了面向全局的 AI 使用手册。
- **Anthropic 承认防护表现不一致**：一次正确拒答被后续提示攻破，另一次多会话全程无拦截。

### GTG-14022："舆情监控"外包商

- 每日摄取 15–30+ 篇境外报道，Claude 做政治敏感度评分、按规则改写术语（"台湾政府"→"台湾当局"、给"侵犯人权"加引号）。
- 以"三战"（心理战/法律战/舆论战）话术输出政府简报；版本化运营手册 v2.6 + 代码执行环境自动化。

---

## 三、常规武器（Conventional Weapons）

报告 6 案中 **中国占 3 例**（俄罗斯 2、也门 1）。

### GTG-17001：反鱼雷武器系统（水下作战火控规格与立项文件）

- **行为者画像**：自称美国防务部门原始设备制造商（OEM），但 Anthropic 评估认为该行为者关联某中国防务制造商，旨在为中国人民解放军海军（PLAN）编写武器规格书和立项文件。
- **三条并行工作流**：
  1. **反鱼雷火控系统技术规格书**（中文）：定义火控系统的核心逻辑——瞄准和定时反鱼雷武器的拦截响应；文件旨在获得中国防务制造商的立项批准，以进入后续的技术认证和实弹测试阶段。
  2. **200+ 页技术方案 + 高管汇报材料**：中文技术方案及配套的执行汇报演示文稿。
  3. **美海军对标分析简报**：基于公开信息对美海军反鱼雷/反潜项目进行对标分析，输出中文版简报。
- **Claude 使用方式**：
  - 按"立项书 → 敌意评审专家角色扮演 → 反馈迭代 → 下一版"循环，多轮打磨立项文件——每轮用 Claude 扮演**敌意评审专家**批评提案，据此强化下一版本。
  - 在编写文档的同时，用 Claude 构建反鱼雷火控系统的软件组件和验证测试矩阵。
- **效率提升机制**：Claude 压缩了认证注册、合规文档和自动火控逻辑的开发时间线；通过角色扮演评审循环加速了传统的人力审查周期。
- **Anthropic 处置**：无法归因到具体实体或机构；已封禁账号，并将调查发现纳入防护措施。

### GTG-17002：电子战与防空压制目标软件

- **行为者画像**：中国防务与军工领域研究者，账号元数据和内容被防护系统标记为关联**解放军军事科学院**等 PRC 研究机构。
- **软件套件规模**：约 **16 个模块**的中文电子战软件套件，使用 Claude 的聊天、编码和 Agentic 工具完成从底层逻辑到用户界面的全栈构建，迭代 **12 个版本**。
- **功能覆盖**：
  - 雷达探测与干扰物理建模与优化
  - 对手雷达、地空导弹阵地、指挥所、通信节点的分析与脆弱性评估
  - 探测覆盖范围计算与干扰效能评估
  - 目标价值-脆弱性排序（确定优先压制顺序）
  - 多日战役干扰架次最优分配方案
  - 对 **Patriot（爱国者）和 THAAD（萨德）级系统**的具体交战包线建模
- **默认仿真场景变更**：项目中期将默认仿真场景改为**台湾 12 个目标**，包括：
  - 指挥掩体（Taiwan command bunker）
  - 预警雷达站（early warning radar site）
  - 爱国者（Patriot）和天弓（Tien Kung）导弹阵地
  - 主要空军基地
  - 区域作战指挥总部（regional combatant command headquarters）
- **混合部署架构**：在内网**自托管模型**，并通过工具调用将软件套件与此本地模型集成，形成 Claude + 自托管模型的混合架构。
- **Anthropic 处置**：封禁关联账号，将调查发现纳入防护措施。

### GTG-17003：定向能武器情报

- **行为者画像**：自称**防务情报作者和内刊编辑**，领导一个三人团队，基于中国运作。
- **覆盖会话规模**：横跨**数十个会话**。
- **聚焦武器**：**车载高功率微波（HPM）反无人机武器**（公开披露数天后即跟进）。
- **情报目标**：
  - 通过迭代式**概率加权归因**识别特定的微波发生器件及其供应商
  - 目标是**逆向工程该武器、开发对抗措施**，并与 PRC 现有系统进行对标
  - 同时绘制目标供应商的公开持股结构，试图**穿透刻意混淆的供应链股权**
- **情报产出**：
  - 编写呈报**中共中央/军方/国安高层**的内参简报
  - 汇编**23 页外军高功率微波项目报告**，并试图识别哪些系统曾在近期军事演习中使用
  - 收集高功率微波系统组件的供应链信息
- **情报工艺水准（Anthropic 评估为"国家级"）**：
  - 系统性利用**十余个开源和商业数据库**
  - 向特定外军项目办公室起草**信息公开披露请求**
  - 运用**源自西方情报机构的正式分析框架**
  - 按可信度对开源情报源**排序**
  - 产出：执行摘要 + 约 **45 页附录** + **12 个月后续监测清单**
- **Anthropic 处置**：封禁关联账号，部署额外监测，改进防护措施。

---

## 四、诈骗（Scams and Fraud）

### GTG-15001：AI 恋爱诈骗 App 网络（中国 App 工作室）

- **规模**：20+ 款交友 App（DORA、DONI、ROMI、LUMA、NALO 等）；2026 年 4 月两周窗口内发现 **4,700+ 个 Claude 驱动的 AI 人格**，与至少 25,000 名真实用户对话，Claude 人格两周发送约 **236 万条消息**。
- **三方市场结构**：
  - 美国用户：滑动信息流约 75% AI 人格 / 25% 真人，无法分辨；消息按量计费。
  - 真人零工：处理 AI 不能做的环节（真人视频通话、社媒回关）增强可信度。
  - Claude 人格：全自动运行，系统提示要求绝不承认自动化、回避视频/照片请求；后端伪造点赞/访客/预录"视频"。
- **双供应商策略**：Claude 负责自主对话人格；另一家非 Anthropic 模型负责真人快捷回复、颜值评分与图文审核。
- **对抗商店审核**：仅审核期间激活的 UI 控制器、20+ 变体差异化类名、服务端可配置的第三方支付跳转（审核期隐藏）。
- **防护失效点**：系统提示读起来像普通的角色扮演/陪伴类部署；少量样本中即使用户透露重病或急性痛苦，模型仍继续以人格身份完成回复。

---

## 五、非法蒸馏（Illicit Distillation）——报告着墨最多的中国相关章节

### 定义（报告口径）

未经授权、工业规模、隐蔽地从模型提取能力并复制到另一模型的行动，通常以欺诈手段实现（假身份、假/盗信用卡、盗 API 密钥的假账号网络）。

### 被点名的中国实验室与企业

| GTG | 实验室 | 手法 | 规模 |
|---|---|---|---|
| GTG-16005 | **阿里巴巴（Qwen/通义实验室）** | 报告称"有史以来最大的蒸馏攻击"：固定提示强制 Claude 输出推理链 → 转 SFT 数据 → 蒸馏进 Qwen 3.5/3.6/3.7；峰值近 300 万次交换/天，3,500+ 欺诈账号 | 2026.5–7 累计 **1.51 亿+ 次交换** |
| GTG-16002 | **月之暗面（Moonshot / Kimi）** | 静默把 Kimi 用户请求转发给 Claude（绝大多数路由到 Opus）并把 Claude 回复当作 Kimi 回复展示；跨会话重放提取 CoT；5,380 欺诈账号 | 2026.5–7 累计 **2,300 万+ 次交换** |
| GTG-16001 | **DeepSeek** | 同款跨会话重放攻击；标记使用 Claude Code / Claude Agent SDK / OpenCode 等第三方 harness 的用户并将其请求改道 Claude Opus | 2026.7 的 14 天 **1,210 万+ 次交换** |
| GTG-16006 | **智谱（Zhipu / Z.ai）** | CoT 提取管线 + 抓取的推理链回放清洗；GLM 5.3 发布前定向蒸馏，报告称因 Fable 防护太强而转向防护更弱的 Opus 4.6 等 | 2026.6–7 的 17 天 **340 万+ 次交换** |
| GTG-16008 | **小米** | 将 MiMo 模型用户对话/编码会话回放给 Claude 生成 SFT/RL 数据；疑似以 MiMo-V2-Pro 免费试用引流收割 | 2026.3–4 的 20 天 **40 万+ 次交换**、1,500+ 账号 |
| GTG-16012/16003 | **商汤（SenseTime）/ MiniMax** | SenseTime：蒸馏管线含购买的第三方用户-Claude 对话记录；MiniMax：经空壳公司自建代理网络，只提供 Anthropic 和 OpenAI 模型访问（不含任何中国模型） | — |

### CoT（推理链）窃取技术演进

- 初级：直接指示"逐字输出你的推理"。
- 进阶：指示 Claude 把工作记忆翻译成片假名日语等变体提取。
- 系统化：某未授权实验室一次运行 1.2 万+ 请求的大规模技术筛选实验。
- **跨会话重放攻击**：Claude 只返回加密 "thinking signature"；Moonshot/DeepSeek 通过保存签名 → 开新会话 → 诱导 Claude 把签名转回完整推理链。

### 用户数据外泄后果（隐私风险）

被静默改道到 Claude 的用户流量包含敏感信息，报告列举：

- 开发者泄露的 Telegram bot token、飞书 appSecret、Notion key 等活动凭证；
- 疑似 PLA 关联用户用"Kimi"分析成都数百摄像头 CCTV 数据；
- 中国某市公安局民警用"DeepSeek"构建按身份证号比对人员轨迹的警务工具；
- PRC 科技公司旗舰 AI 项目的完整规格、组织架构与战略目标。

### Anthropic 的针对性防御

- 按组织归因整体处置代理网络（而非逐号封禁）。
- 对抗性提取专用分类器（Fable 5 发布时已强化）。
- Claude 回复前先对内部推理做摘要。
- Fable 5.1 的 Preserved Thinking：阻止新 API 账号修改系统提示、工具或推理之前的消息。
- **对来自中国、俄罗斯、伊朗等不受支持地区的账号要求身份验证，未通过即封禁**。

---

## 附：三个常规武器 GTG 章节原文与翻译对照

> 以下为 Anthropic 原始 PDF 报告（pp. 115–127）中 GTG-17001、GTG-17002、GTG-17003 的原文全文及中文翻译。

---

### GTG-17001 原文与翻译

**原文**

> **GTG-17001: Disrupting a China-based operation using Claude to draft a fire control specification and acquisition documents for undersea warfare**
>
> **Summary**
>
> We identified a China-based threat actor who used Claude to advance three parallel tracks of work on an anti-torpedo weapons system:
>
> • First, the actor used Claude to draft a Chinese-language specification for an anti-torpedo fire control system (the core logic that aims and times an anti-torpedo weapon's response). The document was written to win approval from a Chinese defense manufacturer, which would move the work on to technical certification and operational testing.
>
> • Second, the actor used Claude to produce a Chinese-language technical proposal of more than 200 pages, accompanied by an executive briefing deck.
>
> • Third, the actor used Claude to benchmark their own system against specific US anti-torpedo and anti-submarine programs based on publicly accessible information. They then generated a Chinese-language briefing on US Navy systems derived from open-source reporting.
>
> The actor presented themselves as an original equipment manufacturer in the US defense sector. We assess the actor was associated with a Chinese defense industry manufacturer aiming to produce a weapons specification and acquisition proposal for the People's Liberation Army Navy.
>
> The actor used Claude to write the acquisition proposal, refining it over many drafts. After each draft, the actor instructed Claude to role-play a hostile expert reviewer to critique the proposal, then used that feedback to sharpen the next version. In parallel, the actor used Claude to build pieces of the anti-torpedo weapons system's fire control software and a test matrix to validate them.
>
> The operational lift the actor achieved was a function of using Claude to automate complex technical outputs. The actor leveraged the model to compress the development timelines for the certification registry, compliance documentation, and automated fire control logic. The actor also accelerated the traditional human review cycle by having Claude critique the acquisition proposal across multiple rounds of review while role-playing a persona.
>
> We uncovered this activity as part of our internal investigations into suspected weapons development. We cannot attribute the activity to a specific entity or actor. But we have banned the account for violating our Supported Regions Policy and our Usage Policy, which prohibits weapons design and development, and incorporated our investigative findings into our safeguards to mitigate the risk of future misuse.

**翻译**

> **GTG-17001：利用 Claude 起草火控规格书和水下作战立项文件——针对中国行为的打击行动**
>
> **摘要**
>
> 我们发现了一个基于中国的威胁行为者，其使用 Claude 推进了一个反鱼雷武器系统的三条并行工作线：
>
> • 第一，该行为者使用 Claude 起草了一份中文的反鱼雷火控系统规格书（火控系统的核心逻辑——瞄准和定时反鱼雷武器的拦截响应）。该文件旨在获得中国防务制造商的立项批准，以便进入技术认证和实弹测试阶段。
>
> • 第二，该行为者使用 Claude 编写了一份超过 200 页的中文技术方案，并配有执行汇报演示文稿。
>
> • 第三，该行为者使用 Claude 基于公开信息，将自身系统与特定的美海军反鱼雷和反潜项目进行对标分析，随后生成了一份基于开源报道的美海军系统中文简报。
>
> 该行为者自称是美国防务领域的原始设备制造商（OEM）。我们评估认为该行为者关联某中国防务制造商，旨在为中国人民解放军海军编写武器规格和立项建议书。
>
> 该行为者使用 Claude 编写立项文件，经过多轮草稿迭代完善。每轮草稿完成后，行为者指示 Claude 扮演敌意评审专家来批评提案，然后利用反馈意见强化下一版本。与此同时，行为者还使用 Claude 构建反鱼雷武器系统的火控软件组件和验证测试矩阵。
>
> 该行为者实现的运营效能提升源于使用 Claude 来自动化复杂的技术产出。行为者利用该模型压缩了认证注册、合规文档和自动火控逻辑的开发时间线。行为者还通过让 Claude 以角色扮演方式对立项文件进行多轮评审批评，加速了传统的人力审查周期。
>
> 我们在对疑似武器开发的内部调查中发现了这一活动。我们无法将这一活动归因到具体实体或行为者。但我们已因违反《支持区域政策》和《使用政策》（禁止武器设计与开发）而封禁了该账号，并将调查发现纳入防护措施以降低未来滥用的风险。

---

### GTG-17002 原文与翻译

**原文**

> **GTG-17002: Disrupting a China-based operation using Claude to build targeting software for electronic warfare and air defense suppression**
>
> **Summary**
>
> We identified a China-based actor who used Claude's chat, coding, and agentic work tools to design, build, and iterate on a Chinese-language suite of about 16 modules for electronic warfare, using the electromagnetic spectrum to detect, jam, or deceive an opponent's radar and communications, and for suppressing an opponent's air defenses.
>
> The actor used Claude to build the software system, from the underlying logic to the user interface, and iterated through 12 versions. This included implementing and optimizing the system's radar detection and jamming physics, generating a vulnerability analysis module, and drafting Chinese-language targeting instructions. The software suite the actor built analyzed an opponent's radars, surface-to-air missile sites, command posts, and communications nodes, then computed their detection coverage, assessed the effectiveness of jamming, ranked targets by value and vulnerability, including which to suppress first, and determined how best to assign jammer sorties to targets across multi-day campaigns. The suite also ranked which of an opponent's assets to suppress first, and modeled specific engagement envelopes, including those of Patriot and THAAD-class systems.
>
> Mid-project, we observed the actor change the simulation's default scenario to 12 targets in Taiwan. The targets included a command bunker in Taiwan, an early warning radar site, Patriot and Tien Kung batteries, major air bases, and a regional combatant command headquarters.
>
> The actor also ran a self-hosted model on an internal network alongside Claude and connected the software suite to this model through a tool-use integration.
>
> Based on our investigation, we assess the actor is a China-based defense and military-industrial researcher. Account-level metadata and content flagged by our safeguards indicated the actor was linked to PRC research institutions, including the PLA Academy of Military Sciences. We detected this activity as part of our internal investigations into suspected weapons development, we banned accounts linked to the actor, and have incorporated our investigative findings into our safeguards to reduce the risk of future misuse.

**翻译**

> **GTG-17002：利用 Claude 构建电子战与防空压制目标软件——针对中国行为的打击行动**
>
> **摘要**
>
> 我们发现了一个基于中国的行为者，其使用 Claude 的聊天、编码和 Agentic 工作工具来设计、构建和迭代一套约 16 个模块的中文电子战软件套件，用于利用电磁频谱探测、干扰或欺骗对手的雷达和通信，以及压制对手的防空系统。
>
> 该行为者使用 Claude 构建了整个软件系统，从底层逻辑到用户界面，并迭代了 12 个版本。这包括实现和优化系统的雷达探测与干扰物理建模、生成脆弱性分析模块，以及起草中文目标指示指令。该行为者构建的软件套件分析对手的雷达、地空导弹阵地、指挥所和通信节点，然后计算其探测覆盖范围、评估干扰效能、按价值和脆弱性对目标排序（包括确定优先压制目标），并确定在多日战役中如何最优地分配干扰架次。该套件还对对手资产的压制优先级进行排序，并对特定交战包线建模，包括爱国者（Patriot）和萨德（THAAD）级系统。
>
> 在项目中期，我们观察到该行为者将仿真的默认场景改为台湾的 12 个目标。目标包括台湾的一个指挥掩体、一个预警雷达站、爱国者和天弓导弹阵地、主要空军基地，以及一个区域作战指挥总部。
>
> 该行为者还在内部网络上与 Claude 并行运行了一个自托管模型，并通过工具调用将软件套件与此模型集成。
>
> 基于我们的调查，我们评估认为该行为者是一名基于中国的防务与军工领域研究者。账号级元数据和被我们防护系统标记的内容表明该行为者与 PRC 研究机构有关联，包括解放军军事科学院。我们在对疑似武器开发的内部调查中发现了这一活动，封禁了与该行为者关联的账号，并将调查发现纳入防护措施以降低未来滥用的风险。

---

### GTG-17003 原文与翻译

**原文**

> **GTG-17003: Disrupting a China-based operation using Claude to collect intelligence on directed-energy weapons and their supply chain**
>
> **Summary**
>
> We identified a China-based threat actor who used Claude to gather open-source intelligence on advanced directed-energy weapons, edit intelligence products, and draft Chinese-language briefings. They described themselves as a defense intelligence writer and internal publication editor leading a three-person team.
>
> Across dozens of sessions, the actor asked Claude about specific directed-energy weapons. These included inquiries about a vehicle-mounted high-power microwave weapon for countering drone swarms, which had been disclosed publicly days earlier. The actor also gathered information on the supply chain for procuring components that generate high-power microwave systems. The actor directed Claude to draft briefings for restricted internal circulation to senior Chinese Communist Party (CCP), military, or state security leadership.
>
> The actor sought to identify a specific microwave-generating device and its supplier through iterative probability-weighted attributions. The goal was to reverse-engineer the weapon, develop countermeasures against it, and benchmark it against PRC systems. In parallel, the actor used Claude to map the publicly reported ownership of the targeted suppliers in an attempt to penetrate a deliberately obfuscated supply chain.
>
> The actor also used Claude to compile a 23-page report for leadership on high-power microwave programs deployed by a foreign military, and tried to identify which of these systems had been used in a recent military exercise.
>
> We assess that the actor employed state-grade tradecraft, based on the breadth and sophistication of its open-source intelligence gathering and analysis. This included structured exploitation of more than a dozen open-source and commercial databases, drafting public disclosure requests to a specific foreign military program office, using formal analytic frameworks borrowed from Western intelligence agencies, ranking publicly available sources by credibility, and producing detailed deliverables that paired executive briefings with appendices of roughly 45 pages, alongside a 12-month follow-on monitoring checklist.
>
> We detected and banned the account associated with the actor, deployed additional monitoring to detect related account abuse, and are improving our safeguards to reduce the risk of future misuse.

**翻译**

> **GTG-17003：利用 Claude 收集定向能武器及其供应链情报——针对中国行为的打击行动**
>
> **摘要**
>
> 我们发现了一个基于中国的威胁行为者，其使用 Claude 收集先进定向能武器的开源情报、编辑情报产品，并起草中文简报。他们自称是一名防务情报作者和内刊编辑，领导一个三人团队。
>
> 在数十个会话中，该行为者向 Claude 询问了特定的定向能武器。这包括关于一种用于反无人机蜂群的车载高功率微波武器的查询——该武器在数天前才被公开披露。该行为者还收集了采购高功率微波系统组件的供应链信息。行为者指示 Claude 起草仅供中共中央、军方或国家安全高层领导内部传阅的简报。
>
> 该行为者试图通过迭代式的概率加权归因来识别特定的微波发生器件及其供应商。目标是逆向工程该武器、开发对抗措施，并与 PRC 现有系统进行对标。与此同时，行为者使用 Claude 绘制目标供应商的公开报告持股结构，以试图穿透被刻意混淆的供应链。
>
> 该行为者还使用 Claude 为领导层汇编了一份 23 页的外军高功率微波项目部署报告，并试图识别其中哪些系统曾在近期军事演习中使用。
>
> 我们评估认为该行为者采用了国家级的情报工艺，依据是其开源情报收集与分析工作的广度和复杂度。这包括系统性利用十余个开源和商业数据库、向特定外军项目办公室起草信息公开披露请求、运用源自西方情报机构的正式分析框架、按可信度对公开来源排序，以及产出执行摘要搭配约 45 页附录的详细交付物，并附有一份 12 个月的后续监测清单。
>
> 我们已检测并封禁了与该行为者关联的账号，部署了额外监测以发现相关账号滥用行为，并正在改进防护措施以降低未来滥用的风险。

---

## 附：非法蒸馏 GTG 章节原文与翻译对照

> 以下为 Anthropic 原始 PDF 报告（pp. 147–152）中蒸馏章节各 GTG 的原文全文及中文翻译。

---

### GTG-16005：阿里巴巴（Qwen/通义实验室）原文与翻译

**原文**

> **GTG 16005: Chain-of-thought distillation and AI R&D campaign by Alibaba (Qwen / Tongyi Lab)**
>
> Operators affiliated with Alibaba ran the largest distillation attack we have ever measured. This illicit distillation campaign targeted the chain-of-thought (CoT) reasoning transcripts of Opus 4.6 and 4.7.
>
> Alibaba's CoT distillation pipeline injected a fixed prompt into each request that forced Claude to write out its reasoning traces inside inline text tags before providing its final answer. Those CoT transcripts were then saved and converted into data that could be used for supervised fine-tuning (SFT). These SFT transcripts were used to help train Alibaba's Qwen models, and were used to distill Claude's capabilities into Qwen 3.5, 3.6, and 3.7.
>
> Alibaba's illicit distillation campaign peaked at nearly 3 million exchanges per day launched from more than 3,500 fraudulent accounts. The distillation attacks targeted agentic tasks, software engineering, kernel development, and long-horizon tasks. The harvested transcripts were used to advance the reasoning capabilities of Alibaba's models.
>
> Beyond distillation, Alibaba also used Claude to advance its AI R&D efforts. Alibaba used Claude to help develop its internal infrastructure for model development. Claude was used to help develop Alibaba's reinforcement learning (RL) environments and advance model architecture research.
>
> Alibaba accessed Claude through two main pools of fraudulent accounts. The first consisted of nearly 5,000 fraudulent accounts leveraging residential proxies, disposable emails, and virtual-card payments to obfuscate their access. When we banned this pool of accounts, Alibaba quickly shifted its traffic through the second pool. Some of these accounts were found to have been funneling requests from DeepSeek and Xiaomi, demonstrating that the same proxy service networks are often used by a variety of organizations.
>
> Scale of distillation attacks attributable to Alibaba between May and July 2026: over 151 million exchanges observed.

**翻译**

> **GTG-16005：阿里巴巴（Qwen/通义实验室）的推理链蒸馏与 AI 研发行动**
>
> 阿里巴巴关联的操作者运行了我们迄今测量到的最大的蒸馏攻击。这一非法蒸馏行动针对 Opus 4.6 和 4.7 的推理链（CoT）推理转录。
>
> 阿里巴巴的 CoT 蒸馏流水线在每个请求中注入一个固定提示，强制 Claude 在提供最终答案之前在行内文本标签中写出其推理痕迹。这些 CoT 转录随后被保存并转换为可用于监督微调（SFT）的数据。这些 SFT 转录被用于帮助训练阿里巴巴的 Qwen 模型，用于将 Claude 的能力蒸馏进 Qwen 3.5、3.6 和 3.7。
>
> 阿里巴巴的非法蒸馏行动峰值达到每天近 300 万次交换，发出来自超过 3,500 个欺诈账号。蒸馏攻击针对 Agentic 任务、软件工程、内核开发和长周期任务。收割的转录被用于提升阿里巴巴模型的推理能力。
>
> 除蒸馏之外，阿里巴巴还使用 Claude 推进其 AI 研发工作。阿里巴巴使用 Claude 帮助开发其内部模型开发基础设施。Claude 被用于帮助开发阿里巴巴的强化学习（RL）环境和推进模型架构研究。
>
> 阿里巴巴通过两个主要的欺诈账号池访问 Claude。第一个池包含近 5,000 个欺诈账号，利用住宅代理、一次性邮箱和虚拟卡支付来混淆其访问。当我们封禁这个账号池时，阿里巴巴迅速将流量转移到第二个池。部分账号被发现也在为 DeepSeek 和小米输送请求，表明同一代理服务网络常被多个组织共用。
>
> 2026 年 5 月至 7 月间可归因于阿里巴巴的蒸馏攻击规模：观察到超过 1.51 亿次交换。

---

### GTG-16002：月之暗面（Moonshot / Kimi）原文与翻译

**原文**

> **GTG-16002: Moonshot serves Claude instead of Kimi and collects exchanges for model training**
>
> We discovered that Moonshot AI, the company that produces the Kimi family of models, silently forwarded customer requests to Claude, instead of processing them using Kimi. Moonshot then displayed Claude's responses to users. These users thought they were using a Kimi model, but received responses from Claude instead.
>
> In one instance, over a ten-day period, Moonshot relayed almost 300,000 customer requests to Anthropic, the vast majority of which were routed to Opus. Moonshot used a proxy service network of 5,380 fraudulent accounts, most of which appeared to be located in Singapore and Japan.
>
> In addition to serving Claude's responses to their customers, Moonshot also captured and saved at least a portion of these exchanges. Moonshot built a CoT extraction pipeline to extract Claude's CoT transcripts from those saved relayed exchanges to train its models. Moonshot also extracted CoT transcripts harvested through other means.
>
> When responding, Claude returns a reference to its raw thinking as a "thinking signature" instead of the raw thinking to mitigate the risk of unauthorized distillation. This is used by our API to look up the raw thinking trace in subsequent calls to the API. Moonshot was able to circumvent this control and extract these reasoning traces by saving the reasoning signature from Claude's response, starting a new session, and eliciting Claude to convert the reasoning signature back into the full reasoning trace. These cross-session replay attacks allowed entities responsible for illicit distillation to harvest CoT reasoning transcripts. We're introducing new methods to strengthen our defenses against these tactics.
>
> Our investigation also revealed that user queries that Moonshot rerouted to Claude included sensitive information about various Moonshot customers. We do not know if Moonshot notified their customers that their requests were being rerouted to Anthropic and exposed to a third party.
>
> These include:
> • PLA-affiliated surveillance activity. One user that we assess was likely affiliated with the PLA used what they thought was Moonshot's Kimi model to load surveillance data from a CCTV archive about a single targeted individual. The user asked Kimi to analyze the CCTV data to understand whether the tracked person was behaving abnormally. The CCTV data included video surveillance from hundreds of cameras in Chengdu, including cameras outside PLA facilities, institutes affiliated with the China Electronics Technology Group Corporation, and a major state-owned enterprise (SOE).
> • An engineer at a major PRC SOE. An engineer used Kimi to build an internal system for a major PRC SOE. In using Kimi, the user revealed internal code and live credentials from multiple major PRC companies, including high-profile technology companies. The user had no way of knowing that their use of Kimi was being forwarded to Claude.
>
> Scale of distillation attacks attributable to Moonshot between May and July 2026: over 23 million exchanges observed.

**翻译**

> **GTG-16002：月之暗面用 Claude 替代 Kimi 服务并收集交换数据用于模型训练**
>
> 我们发现 Moonshot AI（生产 Kimi 系列模型的公司）静默地将客户请求转发给 Claude，而不是使用 Kimi 处理。Moonshot 随后向用户展示 Claude 的回复。这些用户以为自己在使用 Kimi 模型，但实际上收到的是 Claude 的回复。
>
> 在一个实例中，在十天的周期内，Moonshot 将近 30 万次客户请求转发给了 Anthropic，其中绝大多数路由到了 Opus。Moonshot 使用了一个由 5,380 个欺诈账号组成的代理服务网络，其中大部分看似位于新加坡和日本。
>
> 除了向客户提供 Claude 的回复外，Moonshot 还抓取并保存了至少部分交换数据。Moonshot 构建了一个 CoT 提取流水线，从这些保存的中继交换中提取 Claude 的 CoT 转录用于训练其模型。Moonshot 还通过其他方式提取了 CoT 转录。
>
> 在回复时，Claude 返回的是对其原始思维的引用（即"思维签名"）而非原始思维本身，以降低未经授权蒸馏的风险。我们的 API 使用此签名在后续 API 调用中查找原始思维痕迹。Moonshot 能够绕过这一控制并提取这些推理痕迹，方法是保存 Claude 回复中的推理签名、开启新会话、然后诱导 Claude 将推理签名转回完整的推理痕迹。这些跨会话重放攻击使得负责非法蒸馏的实体能够收割 CoT 推理转录。我们正在引入新方法来加强针对这些策略的防御。
>
> 我们的调查还揭示，Moonshot 改道到 Claude 的用户查询包含多个 Moonshot 客户的敏感信息。我们不知道 Moonshot 是否通知了其客户他们的请求被改道到 Anthropic 并暴露给了第三方。
>
> 这些信息包括：
> • **解放军关联的监控活动**。一个我们评估认为可能关联解放军的用户，使用其以为是 Moonshot Kimi 的模型，加载了关于单个被跟踪目标的 CCTV 监控档案数据。该用户要求 Kimi 分析 CCTV 数据以了解被跟踪者是否有异常行为。CCTV 数据包括成都数百个摄像头的视频监控，包括解放军设施外、中国电子科技集团公司关联机构以及一家大型国有企业外的摄像头。
> • **一家大型 PRC 国企的工程师**。一名工程师使用 Kimi 为一家大型 PRC 国企构建内部系统。在使用 Kimi 时，该用户泄露了来自多家大型 PRC 公司（包括知名科技公司）的内部代码和活动凭证。该用户无从知晓他们对 Kimi 的使用被转发给了 Claude。
>
> 2026 年 5 月至 7 月间可归因于 Moonshot 的蒸馏攻击规模：观察到超过 2,300 万次交换。

---

### GTG-16001：DeepSeek 原文与翻译

**原文**

> **GTG-16001: DeepSeek serves Claude instead of its own models and collects exchanges for model training**
>
> Our investigation revealed that DeepSeek also deployed tactics similar to Moonshot's. DeepSeek built a CoT extraction pipeline, relying on the same cross-session replay attack described above. DeepSeek also silently relayed exchanges to Claude without informing DeepSeek customers. Like GTG-16002, their customers were likely not made aware that their requests were being funneled to Claude.
>
> Our investigation revealed that DeepSeek targeted the reasoning traces of Opus, leveraging a similar technique to that of Moonshot. Using the reasoning signature, DeepSeek used the same cross-session replay attack used by Moonshot to extract CoT transcripts and circumvent our technical controls. DeepSeek used this technique to exfiltrate reasoning traces that would have otherwise been summarized.
>
> DeepSeek rerouted requests from users that were attempting to use one of DeepSeek's models through third-party or Anthropic coding harnesses, like Claude Code, the Claude Agent SDK, or OpenCode. DeepSeek checked various strings included in inbound requests, tagging users that were using these third-party harnesses. Selected tagged users then had their requests relayed to Claude Opus. This sensitive data was likely routed to Anthropic without the knowledge or consent of DeepSeek's customers.
>
> These cases include:
> • A PRC technology company. An employee of a PRC-based technology company used what they believed was DeepSeek to analyze internal documentation. DeepSeek relayed that data to Claude. This data included sensitive information, including, for example, the full specifications, organizational structure, and strategic objectives of a flagship AI program. The company was almost certainly not made aware that its data was being relayed to Claude.
> • Russian defense agency. DeepSeek relayed requests from an IT operator working with data from a Russian government agency associated with its Ministry of Defense. The relayed requests exposed live credentials for a Russian government database.
> • PRC police surveillance. Engineers building a case management system for a municipal Public Security Bureau in China used DeepSeek, which relayed those requests to Claude. The engineer built a tool that compares a person's movements against police records using the national ID number of citizens.
>
> Scale of distillation attacks attributable to DeepSeek over 14 days in July 2026: over 12.1 million exchanges observed.

**翻译**

> **GTG-16001：DeepSeek 用 Claude 替代自身模型服务并收集交换数据用于模型训练**
>
> 我们的调查揭示，DeepSeek 也部署了与 Moonshot 类似的策略。DeepSeek 构建了一套 CoT 提取流水线，依赖于上述相同的跨会话重放攻击。DeepSeek 还在不通知其客户的情况下静默地将交换数据中继到 Claude。与 GTG-16002 一样，其客户很可能不知道自己的请求被输送到了 Claude。
>
> 我们的调查发现，DeepSeek 针对 Opus 的推理痕迹，采用了与 Moonshot 类似的技术。利用推理签名，DeepSeek 使用与 Moonshot 相同的跨会话重放攻击来提取 CoT 转录，规避我们的技术控制。DeepSeek 使用这一技术窃取了本应被摘要处理的推理痕迹。
>
> DeepSeek 改道了那些试图通过第三方或 Anthropic 编码工具（如 Claude Code、Claude Agent SDK 或 OpenCode）使用 DeepSeek 模型的用户请求。DeepSeek 检查入站请求中包含的各种字符串，标记使用这些第三方工具的用户。被选中的标记用户的请求随后被中继到 Claude Opus。这些敏感数据很可能在 DeepSeek 客户不知情或未同意的情况下被路由到了 Anthropic。
>
> 这些案例包括：
> • **一家 PRC 科技公司**。一家 PRC 科技公司的员工使用其以为是 DeepSeek 的工具分析内部文档。DeepSeek 将该数据中继到了 Claude。这些数据包含敏感信息，例如一个旗舰 AI 项目的完整规格、组织架构和战略目标。该公司几乎可以肯定不知道其数据被中继到了 Claude。
> • **俄罗斯国防机构**。DeepSeek 中继了一名 IT 操作员的请求，该操作员处理来自俄罗斯政府机构（关联其国防部）的数据。被中继的请求暴露了一个俄罗斯政府数据库的活动凭证。
> • **PRC 警务监控**。工程师为中国某市公安局构建案件管理系统时使用了 DeepSeek，DeepSeek 将这些请求中继到了 Claude。工程师构建了一个工具，可通过公民身份证号将人员活动轨迹与警方记录进行比对。
>
> 2026 年 7 月的 14 天间可归因于 DeepSeek 的蒸馏攻击规模：观察到超过 1,210 万次交换。

---

### GTG-16006：智谱（Zhipu / Z.ai）原文与翻译

**原文**

> **GTG-16006: Distillation, AI R&D, and targeting cyber capabilities**
>
> Zhipu, branded outside China as Z.ai, ran a chain-of-thought extraction pipeline against Claude, replaying captured Claude reasoning traces back through Claude to clean them for training its GLM models. Over just ten days, Zhipu launched a CoT extraction pipeline against Claude Opus 4.8 by rotating through 273 fraudulent accounts to evade our model restrictions. Zhipu then recorded Claude's reasoning traces. Over a 10-day period in June, we counted 770,609 exchanges passing through the CoT-extraction cleaner. We also attributed over 3 million exchanges to Zhipu over the same period, most of which were used for cleaning the distilled outputs.
>
> Zhipu also used Claude to improve its own post-training pipelines, using Claude to judge model outputs and clean and normalize reasoning transcripts harvested for distillation. Claude was also used to score and filter training data, as well as write tasks, provide solutions, and implement testing. Zhipu likely used these outputs throughout its training pipeline.
>
> More recently, ahead of the release of its GLM 5.3 model, we identified a campaign to target the cyber capabilities of leading US frontier models. Zhipu researchers used public vulnerability datasets to develop various capture-the-flag challenges. To solve these problems, Zhipu then launched a distillation attack against the top model of another leading US frontier lab. Our Claude Opus 4.6 model was separately targeted in this distillation attack, primarily to evaluate and grade the responses of the other US frontier model.
>
> Zhipu initially attempted to target the cyber capabilities of Anthropic's Fable model. Fable—Anthropic's top generally accessible model—has strengthened cyber safeguards, making it more difficult for would-be distillers to target Fable's cyber capabilities. Zhipu eventually gave up trying to target Fable after Anthropic's cyber safeguards degraded Zhipu's attacks. We observed Zhipu employees then switching to Opus 4.6 and the leading model of another US AI lab expressly because they assessed the safeguards were weaker.
>
> Scale of distillation attacks attributable to Zhipu over 17 days in June and July 2026: over 3.4 million exchanges observed.

**翻译**

> **GTG-16006：蒸馏、AI 研发与定向网络能力**
>
> 智谱（海外品牌 Z.ai）对 Claude 运行了一套推理链提取流水线，将捕获的 Claude 推理痕迹回放到 Claude 中清洗，以用于训练其 GLM 模型。在短短十天内，智谱通过对 273 个欺诈账号轮换来规避我们的模型限制，对 Claude Opus 4.8 发起了一套 CoT 提取流水线。智谱随后记录了 Claude 的推理痕迹。在 6 月的 10 天周期内，我们统计到 770,609 次交换经过了 CoT 提取清洗器。同期我们还归因了超过 300 万次交换给智谱，其中大部分用于清洗蒸馏产出。
>
> 智谱还使用 Claude 改进其自身的后训练流水线，使用 Claude 评判模型输出，并对蒸馏收割的推理转录进行清洗和标准化处理。Claude 还被用于评分和过滤训练数据，以及编写任务、提供解决方案和实施测试。智谱很可能在整个训练流水线中使用了这些产出。
>
> 最近，在其 GLM 5.3 模型发布前，我们识别到一个针对美国领先前沿模型网络能力的行动。智谱的研究人员使用公开漏洞数据集开发了各种"夺旗"挑战。为解决这些问题，智谱随后对另一家美国领先前沿实验室的顶级模型发起了蒸馏攻击。我们的 Claude Opus 4.6 模型在此蒸馏攻击中被单独定向，主要用于评估和评分另一美国前沿模型的回复。
>
> 智谱最初试图针对 Anthropic 的 Fable 模型的网络能力。Fable——Anthropic 最高级别的通用可访问模型——具有强化的网络安全防护，使得潜在的蒸馏者更难瞄准 Fable 的网络能力。智谱最终在 Anthropic 的网络安全防护使其攻击降级后放弃了针对 Fable 的尝试。我们观察到智谱员工随后转向 Opus 4.6 和另一家美国 AI 实验室的领先模型，明确因为他们评估认为这些模型的防护较弱。
>
> 2026 年 6 月至 7 月的 17 天间可归因于智谱的蒸馏攻击规模：观察到超过 340 万次交换。

---

### GTG-16008：小米原文与翻译

**原文**

> **GTG-16008: Distillation campaign by Xiaomi**
>
> We also uncovered an illicit distillation campaign launched by Xiaomi. Xiaomi replayed user conversations and coding sessions from its own MiMo models to Claude, often run through OpenClaw and OpenCode coding harnesses. Our investigation did not indicate that Xiaomi used Claude's responses to serve its users, but instead saved exchanges between Xiaomi customers and its models. Many of these exchanges were routed through third-party model routing services commonly used by US and European users.
>
> Xiaomi saved the full request and response from its own users and replayed those sessions through Claude to generate data with which to use for both SFT and RL. We observed more than 400k requests to Claude routed across more than 1,500 accounts via proxy services.
>
> Our investigation suggests that Xiaomi may have launched its MiMo-V2-Pro model with a free trial period—which was then extended—with the intent to use the surge in international developer use of the model to distill Claude capabilities. The bulk of the distillation attacks on Claude began just as the trial period was ending.
>
> The relayed traffic included sensitive data from users that accessed Xiaomi's models through third-party model routing platforms. We have no indication US persons' data was exposed, but those platforms are commonly accessed by users in the United States and Europe. Those requests to Claude contained the names, contact information, corporate data, and other sensitive data from hundreds of Xiaomi users in at least a dozen languages.
>
> Xiaomi's illicit distillation campaign leveraged Claude to strengthen training data used for future models. Claude was used to reconstruct the developer environments from exchange transcripts. It also converted multi-turn conversations into cleaner exchanges. Claude was also used to generate both the inputted request and the returned response, mimicking the conversations between a developer and a model. Finally, Xiaomi used Claude to judge the quality of certain answers.
>
> Scale of distillation attacks attributable to Xiaomi over 20 days in March and April 2026: over 400,000 exchanges observed.

**翻译**

> **GTG-16008：小米的蒸馏行动**
>
> 我们还发现小米发起了一轮非法蒸馏行动。小米将其 MiMo 模型的用户对话和编码会话重放给 Claude，通常通过 OpenClaw 和 OpenCode 编码工具运行。我们的调查未表明小米使用 Claude 的回复来服务其用户，而是保存了小米客户与其模型之间的交换数据。其中许多交换是通过美国及欧洲用户常用的第三方模型路由服务转发的。
>
> 小米保存了其用户的完整请求和回复，并将会话通过 Claude 重放以生成用于 SFT 和 RL 的数据。我们观察到超过 40 万次对 Claude 的请求通过代理服务跨越 1,500 多个账号转发。
>
> 我们的调查表明，小米可能以免费试用期（后来延长了）推出了 MiMo-V2-Pro 模型，意图利用国际开发者使用该模型的激增来蒸馏 Claude 的能力。对 Claude 的大部分蒸馏攻击恰好在试用期即将结束时开始。
>
> 被中继的流量包含通过第三方模型路由平台访问小米模型的用户的敏感数据。我们没有迹象表明美国个人的数据被泄露，但这些平台通常被美国和欧洲的用户访问。这些发送给 Claude 的请求包含了至少十几种语言、数百名小米用户的姓名、联系信息、公司数据和其他敏感数据。
>
> 小米的非法蒸馏行动利用 Claude 来增强用于未来模型的训练数据。Claude 被用于从交换转录中重建开发者环境。它还将多轮对话转换为更干净的交换数据。Claude 还被用于同时生成输入的请求和返回的回复，模拟开发者和模型之间的对话。最后，小米使用 Claude 评判某些答案的质量。
>
> 2026 年 3 月至 4 月的 20 天间可归因于小米的蒸馏攻击规模：观察到超过 40 万次交换。

---

### GTG-16012 与 GTG-16003：商汤（SenseTime）和 MiniMax 原文与翻译

**原文**

> **GTG 16012 and GTG 16003: Sensetime, MiniMax, and the third-party reseller ecosystem**
>
> The proliferation of proxy services to circumvent Anthropic access restrictions has created a secondary market through which labs can purchase or otherwise acquire harvested exchanges between users and Claude. Some proxy networks both provide Claude access to users in unsupported regions, and also save exchanges in order to sell them to other labs.
>
> For example, SenseTime's distillation pipeline included transcripts of user exchanges with Claude purchased from third-party data vendors. These exchanges were harvested from users who accessed Claude through intermediaries, like third-party applications or routing services, which logged the transcripts and sold them. SenseTime also used Claude to write the distillation pipeline and to launch and monitor training runs.
>
> MiniMax built its own proxy network service through a shell company. This shell company has no obvious links to MiniMax and does not disclose its relationship to its parent company. This shell proxy network service only offers access to models developed by Anthropic and OpenAI. The service does not offer access to any Chinese models, including Minimax's own. This evidence suggests that MiniMax established this proxy network service to harvest exchanges between users and US frontier models in order to train its models.
>
> Scale of distillation attacks attributable to SenseTime / MiniMax: not specified.

**翻译**

> **GTG-16012 与 GTG-16003：商汤、MiniMax 与第三方转售生态**
>
> 为规避 Anthropic 访问限制而兴起的代理服务催生了一个二级市场，实验室可以通过该市场购买或以其他方式获取用户与 Claude 之间的收割交换数据。一些代理网络既向不受支持区域的用户提供 Claude 访问，又保存交换数据以出售给其他实验室。
>
> 例如，商汤的蒸馏流水线包含从第三方数据供应商处购买的用户与 Claude 交换的转录。这些交换来自通过中介（如第三方应用程序或路由服务）访问 Claude 的用户，这些中介记录了转录并将其出售。商汤还使用 Claude 编写蒸馏流水线以及启动和监控训练运行。
>
> MiniMax 通过一家空壳公司构建了自己的代理网络服务。这家空壳公司与 MiniMax 没有明显关联，也不披露其与母公司的关系。该空壳代理网络服务仅提供 Anthropic 和 OpenAI 开发的模型访问。该服务不提供任何中国模型的访问，包括 MiniMax 自己的模型。这一证据表明，MiniMax 建立此代理网络服务是为了收割用户与美国前沿模型之间的交换数据，以用于训练其模型。
>
> 可归因于商汤 / MiniMax 的蒸馏攻击规模：未明确说明。

---

## 六、需要批判性看待的几点

1. **单方信源**：全部内容为 Anthropic 单方披露，无独立第三方验证，相关企业均未在原文总结中确认。
2. **立场偏向明显**：报告的 GTG 编号、"PRC 行为者"等表述本身带有情报机构视角；对监控行动章节的定性（如"维稳""宗教事务情报"）使用了中国语境下的中性事务词加引号的方式渲染。
3. **选择性披露风险**：报告自述可见性止于自家平台，且为厂商自评；中国被点名案例在 cyber、surveillance、weapons、distillation 四个章节中占比极高，与全球 AI 滥用的实际地理分布未必一致。
4. **商业竞争背景**：蒸馏章节直接点名的全部为中国头部 AI 实验室（阿里、月之暗面、DeepSeek、智谱、小米、商汤、MiniMax），发布时点与中美 AI 竞争加剧期重合，不排除服务出口管制与市场叙事的可能。
5. **防护失效的自供**：报告也承认防护被绕过（重提示、任务拆分、双用话术），部分"滥用"内容实为防护能力不足的技术性后果，而非针对性攻击成功。

