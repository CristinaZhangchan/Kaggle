# 面试准备文档 — Campaign Funnel 项目

> 用法：先看第 1-5 部分把项目和代码彻底搞懂（中文大白话），
> 第 6 部分是面试问答（英文问 + 英文答 + 中文解释），
> 第 7 部分是答不上来时的救场话术。

---

## 第 1 部分：这个项目到底在干什么（电梯陈述）

如果面试官说 "Tell me about this project"，你就背这段（英文在第 6 部分 Q1）：

一家广告代理公司，帮很多客户在 **Google / Meta / TikTok / LinkedIn** 上投广告。
广告（campaign）按"营销漏斗"分两种目的：

- **Top-funnel（漏斗顶部）**：打广告给**陌生人**看，目的是让更多人**认识品牌**（拉新、曝光）。
- **Bottom-funnel（漏斗底部）**：打广告给**已经对品牌有兴趣的人**，目的是让他们**直接下单**（转化）。

**问题**：客户没有统一标注每个广告属于哪种。系统里根本没有 `funnel_level` 这个字段。
有些客户广告起名能看出来，有些完全看不出来。

**我的任务**：给数据集里**所有广告**自动打上 `top` / `bottom` 标签。

**做法分两步（这是关键，叫弱监督 weak supervision）**：

1. **Stage A**：不训练模型，先用"规则/线索"给**一部分**广告打上**比较有把握**的标签。
2. **Stage B**：用 Stage A 这批标签当"教材"，训练一个机器学习模型，再让模型给**全部**广告打标签。

**为什么要这么绕？** 因为一开始一个标签都没有（没有"标准答案"）。机器学习模型必须先有标准答案才能学。
所以 Stage A 的作用就是"先人工造出一批标准答案"。

**Part 2** 是后续：用打好的标签去研究 —— 一个客户偏 top 还是偏 bottom，
能不能从他网站的流量数据里看出端倪。

---

## 第 2 部分：Task 1 概念讲解（大白话）

### 什么是"营销漏斗"（marketing funnel）？

想象一个漏斗。最上面口大，最下面口小：

- **漏斗顶部**：很多人——他们还不认识你的品牌。Top-funnel 广告就是去够到这些陌生人。
- **漏斗底部**：少数人——他们已经认识你、有兴趣了。Bottom-funnel 广告就是推这些人下单。

为什么要区分？因为这两种广告的**优化方式、预算、汇报口径完全不同**。
你不能用"卖货"的标准去考核一个只想"刷曝光"的广告。

### 怎么判断一个广告是 top 还是 bottom？数据里有哪些线索？

| 线索字段 | top-funnel 长这样 | bottom-funnel 长这样 |
|---|---|---|
| `audience_type`（受众） | broad / affinity（广泛、兴趣） | remarketing / customer_list（再营销、老客户名单）|
| `objective`（广告目标） | Awareness / Reach（曝光、触达）| Conversions / Sales（转化、销售）|
| `optimization_event`（优化事件）| Impressions / Views（看到、观看）| Purchase / Lead（购买、留资）|
| `ad_format`（创意格式）| Video（视频，适合讲故事）| Lead form / Dynamic product ad（留资表单、动态商品广告）|
| `campaign_name`（广告名字）| 含 "awareness/launch/prospecting" | 含 "retarget/conversion/branded" |
| 行为指标 | CPM 高、CTR 低、转化率低、没有 ROAS | CTR 高、转化率高、有 ROAS |

> **CPM** = 每一千次曝光的成本。**CTR** = 点击率。**ROAS** = 广告花 1 块钱赚回几块钱。
> **ROAS 为空** 本身就是线索——说明这个广告根本没在追踪"转化价值"，多半是 top-funnel。

### 为什么这是个难题，不能写一条规则搞定？

因为线索**不完整也不一致**：
- 有的广告名字乱起，看不出来。
- 有的受众类型（`in_market`、`lookalike_small`）模棱两可。
- 不同平台字段写法不一样。

单条规则覆盖不全。所以才需要"先用规则造一批样本，再用模型学会更泛化的规律"。

---

## 第 3 部分：Task 1 代码讲解

一共三个 notebook，按顺序跑：`01_eda` → `02_stage_a_labels` → `03_stage_b_model`。

### 3.1 `01_eda.ipynb` —— 探索数据

**目的**：动手做模型前，先摸清数据长什么样、有没有坑。

逐块讲：

1. **读数据**：`pd.read_csv` 读进来，4546 行 / 637 个广告 / 120 个客户 / 4 个平台 / 12 个月。
   - 关键认知：**一个广告每活跃一个月就有一行**。所以 4546 行 ≠ 4546 个广告。
2. **每个广告活跃几个月**：`groupby("campaign_id").month.nunique()`，画柱状图看分布。
3. **缺失值检查**：`df.isna().mean()`。重点看 `cpc_usd / conversion_rate / roas` 为什么是空。
   - `roas` 空 = 没追踪转化价值。我专门算了"不同受众类型下 roas 为空的比例"——证明这是个有用的线索。
4. **平台/客户分布**：每个平台多少广告、每个客户多少广告，柱状图 + 直方图。
5. **数值分布**：impressions / spend 这些是**长尾**（少数广告超大），所以画图用 `np.log10` 取对数，否则全挤在一起看不清。
6. **漏斗相关探索**（最重要）：
   - `audience_type` vs `ctr / conversion_rate / cpm` 的**箱线图（boxplot）**——验证"老客户受众转化率高"。
   - `objective × optimization_event` 的**热力图（heatmap）**——看这两个字段怎么搭配。
   - `cpm vs ctr` 散点图，按受众着色——看不同受众能不能在图上分开。
7. **广告名字文本探索**：把名字拆成单词，数词频，找漏斗关键词。
8. **客户级汇总**：为 Part 2 提前准备，算每个客户的总花费、广告数。

**EDA 的结论**（面试要能说出来）：数据小（几百个广告）→ 用线性模型/树模型，不用深度学习；
有多行 → 建模前要先聚合到广告级别；线索很多；`roas` 是否为空是个好特征；
同一个客户的广告命名风格像 → 交叉验证必须按客户切分（防数据泄漏）。

### 3.2 `02_stage_a_labels.ipynb` —— 用规则造标签

**目的**：不用模型，用 5 路"线索投票"给广告打标签，并给每个标签标上**可信度等级**。

逐块讲：

1. **聚合到广告级别**：配置字段（受众、目标…）每个广告不变，所以 `drop_duplicates("campaign_id")`。
2. **定义 5 个信号函数**，每个函数看一个字段，返回 `top` / `bottom` / `None`（看不出来）：
   - `audience_signal`：broad/affinity/lookalike_large → top；remarketing/customer_list → bottom。
   - `objective_signal`：名字含 awareness/reach → top；含 conversion/sales/lead → bottom。
   - `optim_signal`：优化 impression/view → top；优化 purchase/lead → bottom。
   - `format_signal`：video → top；lead form / dynamic product → bottom。
   - `name_signal`：广告名字里有没有漏斗关键词。
3. **投票**：每个广告 5 个信号投票。`top` 票多就是 top，`bottom` 票多就是 bottom，**平票就算 None**。
4. **可信度分级**（`confidence`）：
   - `high`：3+ 个信号且**全部一致**（或 4+ 个且 75% 一致）。
   - `medium`：2+ 个信号且多数一致。
   - `low`：只有 1 个信号，或信号打架。
5. **质量检查**（面试爱问）：
   - **信号两两一致率热力图**：任意两个信号在都开火的广告上，意见一致的比例。一致率高 = 信号互相印证、靠谱。
   - 每个平台的标签分布——sanity check，看有没有哪个平台全是一种标签（不正常）。
   - 随机抽 10 个 high-confidence 的 top 和 bottom，**人眼看一遍**对不对。
   - 看看那些 None（没标上）的广告长什么样。
6. **保存** `campaigns_stage_a.csv`。

**Stage A 结果**：标上了 **91.7%** 的广告（high 53%、medium 32%、low 7%），剩 8% 实在看不出来。

> 重点：只有 **high + medium** 会进 Stage B 当训练教材。low 和没标上的，是留给模型去**预测**的对象。

### 3.3 `03_stage_b_model.ipynb` —— 训练模型

**目的**：用 Stage A 的可信标签训练分类器，再给全部 637 个广告打标签。

逐块讲：

1. **读数据**：Stage A 标签 + 原始月度指标。
2. **聚合到广告级别**：每个广告把 12 个月的指标加总/平均。
   - 比率（CTR 等）我用**总数重新算**（`总点击 / 总曝光`），比"月度比率求平均"更准。
   - `roas_tracked`：这个广告到底有没有追踪 ROAS，做成 0/1 特征。
   - 量级大的字段（impressions/spend）做 `log1p` 对数变换。
3. **划分训练集/预测集**：训练集 = high+medium（540 个广告）；预测对象 = 全部 637 个。
   - `sample_weight`：high 权重 1.0，medium 权重 0.5——告诉模型"高可信的标签更重要"。
4. **特征**三类：
   - 数值：对数后的量级、CTR、转化率、CPM、CPC、预算、活跃月数、`roas_tracked`。
   - 类别：平台、目标、出价策略、优化事件、格式、受众——用 **one-hot 编码**变成 0/1 列。
   - 文本：广告名字用 **TF-IDF** 变成数字（见 Q26）。
5. **管线 1：逻辑回归（Logistic Regression）** + 5 折 **GroupKFold（按 customer_id 切）** 交叉验证。
6. **管线 2：LightGBM**（梯度提升树），同样的交叉验证。
7. **对比**：打印两个模型的 precision/recall/AUC，画混淆矩阵。
8. **选最好的**（AUC 高的），在全部 540 个训练样本上重新训练，预测全部 637 个。
   - 输出 `funnel_level` + `p_bottom`（预测概率）+ `model_confidence`（概率离 0.5 远不远）。
9. **一致性检查**：模型预测和 Stage A 标签对得上吗？对得上说明模型没跑偏。
10. **特征重要性**：看模型最看重哪些特征——验证它学到的东西合理。
11. **保存** `campaigns_labeled.csv`。

**Stage B 结果**：逻辑回归 AUC **0.994**，LightGBM AUC 0.985，选了逻辑回归。
最终 321 个 top / 316 个 bottom。模型和 Stage A 一致率 **97.6%**。

---

## 第 4 部分：Task 2 概念讲解（大白话）

**Part 1** 给每个广告打好了 top/bottom 标签。**Part 2** 换一个数据集：
`traffic.csv`——每个客户每个月的**网站流量**数据（多少人来、从哪来的、来了之后质量如何）。

**核心问题**：一个客户的广告策略（偏 top 还是偏 bottom），
能不能从他网站流量的"样子"里看出来？

**直觉假设**：
- 一个**偏 bottom-funnel** 的客户 → 投的都是"老客户、有意向的人"→ 网站上**直接访问**、
  **品牌词搜索**的人多，跳出率低，转化率高。
- 一个**偏 top-funnel** 的客户 → 一直在拉新 → 网站上**新访客**多、**社交媒体来的**多、
  **非品牌词搜索**多，跳出率高。

**做法**：
1. 算每个"客户×月份"的 **funnel mix（漏斗组合）** = bottom-funnel 花的钱 ÷ 总花费。
2. 跟流量数据 join 起来。
3. 算**相关性（correlation）**，画散点图、热力图、分组柱状图。

### `04_part2_traffic.ipynb` 代码讲解

1. **读 3 个文件**：原始月度广告数据（要花费）、Part 1 的标签、流量数据。
2. **Join**：给每条月度广告行贴上 funnel 标签，按"客户×月×漏斗类型"加总花费。
3. **选 funnel mix 指标**：用 **bottom-funnel 花费占比**（见 Q42 解释为什么）。
4. **处理没广告的月份**：流量表每个客户每月都有行，但有些客户某月没投广告 →
   funnel mix 算不出来。我**直接丢掉这些行**（占 3.9%），因为硬填一个数等于"编造"策略。
5. **流量特征**：把各来源的访问量换成**占比**（÷ 总访问量），这样大小客户可比。
6. **相关性分析**：funnel mix vs 各流量来源占比、vs 流量质量指标，画散点 + 回归线。
7. **相关性热力图**：一张图看全部关系。
8. **客户分组**：把客户按平均 funnel mix 分成 top-heavy / balanced / bottom-heavy 三组，
   对比三组的典型流量画像（柱状图）——这是给非技术的人看的最直观方式。

**Part 2 结果**（相关性很强，因为是合成数据）：
bottom-funnel 占比越高 →
- **正相关**：网站转化率 (+0.87)、直接访问占比 (+0.76)、每次访问页数 (+0.74)、邮件占比 (+0.68)。
- **负相关**：跳出率 (-0.78)、新访客率 (-0.76)、非品牌搜索占比 (-0.70)、社交占比 (-0.62)。

**结论**：是的，funnel 策略在网站流量上留下了清晰的"指纹"。假设全部成立。

---

## 第 5 部分：必须记住的关键数字

| 项目 | 数字 |
|---|---|
| 数据集 | 4546 行月度数据，637 个广告，120 个客户，4 个平台，12 个月 |
| Stage A 覆盖率 | 91.7%（high 53% / medium 32% / low 7%，8% 没标上）|
| Stage B 训练集 | 540 个广告（high+medium）|
| 逻辑回归 AUC | 0.994 |
| LightGBM AUC | 0.985 |
| 最终标签 | 321 top / 316 bottom |
| 模型 vs Stage A 一致率 | 97.6%（high 档 100%，low 档 86%）|
| Part 2 分析行数 | 1384（丢掉 56 个没广告的月份，占 3.9%）|

---

## 第 6 部分：面试问答（50 题）

> 格式：**英文问题** → *答（背英文）* → *解释（看中文）*。
> 英文答案故意写得简单、短，方便你说出口。

### A. 项目概述

**Q1 — Can you walk me through this project?**

*答：* "Sure. A company runs ad campaigns for many clients on Google, Meta, TikTok and LinkedIn.
Each campaign is either *top-funnel* — reaching new people — or *bottom-funnel* — driving sales
from warm audiences. But the clients did not label this. My task was to label every campaign
automatically. I did it in two stages. First, I used simple rules to label the campaigns I was
confident about. Then I trained a machine learning model on those labels and used it to label
all campaigns."

*解释：* 这是你的开场白，背熟。讲清楚"问题是什么 + 两步做法"。慢慢说，不用急。

**Q2 — What was the main challenge here?**

*答：* "The main challenge was that there were no labels at all to start with. A machine
learning model needs labeled examples to learn from. So I had to create the first labels
myself, using rules. This is called weak supervision."

*解释：* 核心难点 = 一开始零标签。模型要学就得有标准答案，所以先自己造。这套方法叫"弱监督"。

**Q3 — What does "funnel level" mean? Why does it matter?**

*答：* "Funnel level is the goal of a campaign. Top-funnel campaigns reach new people who don't
know the brand yet — the goal is awareness. Bottom-funnel campaigns target people who already
showed interest — the goal is to make them buy. It matters because the two types are
optimized, budgeted, and measured in completely different ways."

*解释：* top = 拉新/曝光，bottom = 转化/卖货。重要是因为两种广告的优化、预算、考核标准完全不同。

**Q4 — Why not just write one rule to label everything?**

*答：* "Because the clues are not complete. Some campaign names are unclear. Some audience
types are ambiguous. Different platforms use different wording. One rule cannot cover
everything. A model can learn a more general pattern from many features at once."

*解释：* 单条规则覆盖不全——名字乱、受众模糊、平台写法不一。模型能综合很多特征学到更泛化的规律。

**Q5 — What is your final output?**

*答：* "A file called `campaigns_labeled.csv`. It has every campaign with a predicted funnel
level — top or bottom — a probability, and a confidence level."

*解释：* 最终产出就是这个 CSV：每个广告 + 预测标签 + 概率 + 可信度。

**Q6 — If you had more time, what would you improve?**

*答：* "Two things. First, I would hand-label a small set of campaigns myself, completely
separate from the rules, to use as a real test set. Right now I cannot fully measure accuracy
because I have no ground truth. Second, I would tune the rules in Stage A by looking at the
campaigns the model and the rules disagree on."

*解释：* 改进点：(1) 人工标一小批**独立**样本做真正的测试集（现在没有标准答案，没法真正测准确率）；
(2) 看模型和规则打架的广告，反过来优化规则。

### B. 数据探索（EDA）

**Q7 — What did you find when you explored the data?**

*答：* "The dataset is small — about 637 campaigns. Each campaign has one row per active month,
so I had to aggregate to campaign level before modeling. The numeric values like spend and
impressions are very skewed, so I used a log scale. And I found strong clues for the funnel
label in the audience type, the objective, and the campaign name."

*解释：* 数据小、有多行要聚合、数值偏态要取对数、线索很多。

**Q8 — How big is the dataset?**

*答：* "4546 monthly rows, which is 637 unique campaigns, 120 customers, 4 platforms, over 12
months."

*解释：* 把这几个数字背下来。

**Q9 — Why did you aggregate the data to campaign level?**

*答：* "Because a campaign appears in many rows — one per active month. But the funnel level is
a property of the whole campaign, not of one month. So I combined the monthly rows into one
row per campaign. I summed the totals like spend and impressions, and recomputed the rates."

*解释：* funnel level 是整个广告的属性，不是单月的。所以把多月行合并成一行。总量加总，比率重算。

**Q10 — Why did you use a log scale for some plots?**

*答：* "Because values like impressions and spend are very skewed — a few campaigns are huge
and most are small. On a normal scale everything is squashed together. A log scale spreads
them out so the shape of the distribution is visible."

*解释：* impressions/spend 长尾——少数超大、多数很小。普通坐标全挤一起，取对数后才看得清分布。

**Q11 — What does a missing ROAS value mean?**

*答：* "ROAS is return on ad spend. It is missing when the campaign does not track conversion
value at all. That is not random — it is itself a signal. Campaigns that don't track ROAS are
usually top-funnel. So I turned 'is ROAS tracked' into a feature."

*解释：* ROAS 空 ≠ 随机缺失，而是"这个广告根本不追踪转化价值"——这本身就是 top-funnel 的信号。
我把它做成了一个 0/1 特征。

**Q12 — Did you find any data quality problems?**

*答：* "Nothing serious. Some rates like CPC and conversion rate are null when there were no
clicks — that is expected, not an error. The traffic source counts don't add up to the total
exactly, off by a few sessions, probably rounding. I handled both safely."

*解释：* 没大问题。CPC/转化率为空是因为没点击，正常。流量来源加起来和总数差几个，是四舍五入。

### C. Stage A — 启发式标注

**Q13 — Explain Stage A. Why did you do it this way?**

*答：* "Stage A creates the first labels without a model. I used five separate signals — the
audience type, the objective, the optimization event, the ad format, and the campaign name.
Each signal votes top or bottom. If the votes agree, I assign that label. I needed this
because a model cannot train without labels, and hand-labeling 637 campaigns by hand would be
slow and inconsistent."

*解释：* Stage A 用 5 个信号投票造标签。需要它是因为模型没标签没法训练，纯手工标 637 个又慢又不一致。

**Q14 — What signals did you use, and why those?**

*答：* "Five. Audience type — warm audiences like remarketing mean bottom-funnel. Objective —
'awareness' means top, 'conversion' means bottom. Optimization event — optimizing for
impressions is top, for purchases is bottom. Ad format — video is usually top, lead forms are
bottom. And keywords in the campaign name. I chose them because each one independently relates
to the campaign's goal."

*解释：* 5 个信号，每个都独立地和广告目的相关。能背几个例子就行。

**Q15 — Why not hand-label every campaign yourself?**

*答：* "Three reasons. It is slow. It is inconsistent — I might judge similar campaigns
differently on different days. And it does not scale — if the dataset grows to 100,000
campaigns, hand-labeling is impossible. Rules are fast, consistent, and scale."

*解释：* 手工标：慢、不一致、不可扩展。规则：快、一致、可扩展。

**Q16 — How do you know the heuristic labels are correct?**

*答：* "I can't be 100% sure — and I expect some are wrong. But I built checks. The strongest
one is a pairwise agreement check: I look at how often two independent signals agree on the
same campaign. They agree most of the time, which means the signals confirm each other. I also
spot-checked random labels by eye."

*解释：* 不能 100% 确定，肯定有错的——这是预期之内。但我做了检查：最强的是"两两信号一致率"——
两个独立信号大多数时候意见一致，说明互相印证。还人眼抽查了。

**Q17 — What is the confidence tier? Why do you need it?**

*答：* "Each Stage A label gets a confidence — high, medium, or low — based on how many signals
agree. High means three or more signals all agree. I need it because not all rule-based labels
are equally trustworthy. I only train the model on high and medium labels, and I give high
labels more weight."

*解释：* 按几个信号一致，给标签分 high/medium/low 三档。需要它因为规则标签可信度不一样。
只用 high+medium 训练，high 给更大权重。

**Q18 — How much of the data could Stage A label?**

*答：* "91.7%. About 53% were high confidence, 32% medium, and 7% low. The remaining 8% had no
clear signal at all."

*解释：* 91.7%。high 53%、medium 32%、low 7%，剩 8% 完全看不出来。

**Q19 — What happens if two signals disagree?**

*答：* "It depends on the votes. If there is a clear majority — say three signals say bottom and
one says top — I take the majority and mark it lower confidence. If it is a tie, I assign no
label and leave that campaign for the model to predict later."

*解释：* 看票数。有明显多数就取多数，但降可信度。平票就不给标签，留给模型预测。

**Q20 — Campaign names are useful, but why are they also risky?**

*答：* "They are useful because clients often put words like 'retargeting' or 'awareness' in
the name. They are risky because names are free text — clients can name a campaign anything,
misspell it, or use internal codes. So the name is one signal among five, never used alone."

*解释：* 名字有用是因为常含漏斗关键词；有风险是因为是自由文本，可以乱起、拼错、用内部代号。
所以名字只是 5 个信号之一，绝不单用。

**Q21 — What did you do with campaigns Stage A could not label?**

*答：* "They are not thrown away. They become the prediction targets for the model in Stage B.
The whole point of training a model is to label exactly those campaigns that the rules
could not."

*解释：* 不丢。它们正是 Stage B 模型要预测的对象。训模型的意义就是搞定规则搞不定的那些。

**Q22 — Isn't this circular? Your own labels become the training data.**

*答：* "It is a fair concern. But it is not fully circular, because the model sees more features
than the rules used and learns a smoother, more general pattern. The model can correct some
rule mistakes. And I keep the low-confidence labels out of training so noisy labels don't
teach the model bad patterns. The honest fix would be a separate hand-labeled test set."

*解释：* 这个担心合理。但不算完全循环：模型看到的特征比规则多，学到更平滑泛化的规律，能纠正部分规则错误。
而且低可信标签不进训练。真正彻底的解法是独立的人工测试集。

### D. Stage B — 模型

**Q23 — Explain Stage B.**

*答：* "In Stage B I take the high and medium confidence labels from Stage A — 540 campaigns —
and train a classifier. I built features from numbers, categories, and the campaign name text.
I tried two models, Logistic Regression and LightGBM, and evaluated them with cross-validation.
Then I refit the best model and predicted the funnel level for all 637 campaigns."

*解释：* Stage B：拿 540 个可信标签训分类器，特征 = 数值+类别+文本，试两个模型，交叉验证，
选最好的，预测全部 637 个。

**Q24 — Why Logistic Regression and LightGBM? Why not deep learning?**

*答：* "The dataset is small — only a few hundred campaigns. Deep learning needs much more data
and would just overfit. Logistic Regression is a simple, fast, interpretable baseline. LightGBM
is a tree model that can catch non-linear patterns. Comparing a simple and a stronger model
tells me if the extra complexity is worth it."

*解释：* 数据太小，深度学习会过拟合。逻辑回归简单快可解释，LightGBM 能抓非线性。
拿一个简单的和一个强的对比，看复杂度值不值。

**Q25 — What features did you use?**

*答：* "Three groups. Numeric — like log of spend and impressions, CTR, conversion rate, CPM,
and whether ROAS is tracked. Categorical — platform, objective, audience type, and so on,
turned into 0/1 columns with one-hot encoding. And text — the campaign name, turned into
numbers with TF-IDF."

*解释：* 三组：数值、类别（one-hot）、文本（TF-IDF）。

**Q26 — What is TF-IDF? Why use it on campaign names?**

*答：* "TF-IDF turns text into numbers. TF is term frequency — how often a word appears.
IDF is inverse document frequency — it lowers the weight of words that appear everywhere,
like 'campaign', and raises the weight of rare, meaningful words, like 'retargeting'.
I use it so the model can read the campaign name as features."

*解释：* TF-IDF 把文本变数字。TF = 词出现多频繁；IDF = 到处都有的词（如 campaign）降权，
少见有意义的词（如 retargeting）升权。让模型能"读"广告名。

**Q27 — Why GroupKFold by customer_id? What is data leakage?**

*答：* "Data leakage is when information from the test set sneaks into training, making the
score look better than it really is. Here, one customer names all their campaigns in a similar
style. If some of a customer's campaigns are in training and others in testing, the model can
recognize the style instead of learning the real pattern. GroupKFold keeps all of one
customer's campaigns on the same side of the split, so the test score is honest."

*解释：* 数据泄漏 = 测试集信息偷偷进了训练，分数虚高。这里同一个客户广告命名风格像，
如果一个客户的广告横跨训练和测试，模型会"记风格"而不是学真规律。
GroupKFold 保证一个客户的广告全在同一边，分数才真实。

**Q28 — What is sample_weight and why use it?**

*答：* "Sample weight tells the model how important each training example is. I gave high
confidence labels a weight of 1.0 and medium confidence labels 0.5. So the model trusts the
labels I am more sure about, but still learns a little from the less sure ones."

*解释：* 样本权重 = 告诉模型每个样本多重要。high 给 1.0，medium 给 0.5。
更信可信的标签，但也从不太可信的里学一点。

**Q29 — What is class_weight balanced?**

*答：* "If one class has more examples than the other, the model can get lazy and just predict
the bigger class. class_weight balanced makes the model pay equal attention to both classes
by giving the smaller class more weight. My classes were fairly balanced, but I used it as a
safety measure."

*解释：* 如果一类样本多，模型会偷懒只猜多的那类。class_weight balanced 给少的那类加权，
让模型两类一视同仁。我的数据比较平衡，用它是保险。

**Q30 — Your AUC is 0.99. Isn't that too good? Are you overfitting?**

*答：* "It is a fair worry, so let me explain why I think it is real. First, this is a
cross-validation score, where whole customers were held out — not a training score. Second,
this is synthetic data, so the patterns are cleaner than in the real world. Third, the
features and the Stage A rules overlap a lot — the model is partly learning the same logic
the rules used. So a high score makes sense here. On real, messy data I would expect lower."

*解释：* 担心合理。但：(1) 这是交叉验证分数，整个客户被留出，不是训练分数；
(2) 合成数据，规律比真实世界干净；(3) 特征和规则重叠多，模型部分在学规则的逻辑。
所以高分合理。真实脏数据上会更低。

**Q31 — How did the two models compare? Which did you pick?**

*答：* "Logistic Regression scored slightly higher — AUC 0.994 versus 0.985 for LightGBM. So I
picked Logistic Regression. It is also simpler and easier to explain. When a simple model
matches or beats a complex one, you should take the simple one."

*解释：* 逻辑回归略高（0.994 vs 0.985），所以选它。它还更简单可解释。
简单模型打平或赢复杂模型时，选简单的。

**Q32 — What is ROC-AUC? Why look at it instead of accuracy?**

*答：* "AUC measures how well the model ranks a random bottom-funnel campaign above a random
top-funnel one. It is 0.5 for random guessing and 1.0 for perfect. I look at it because it
does not depend on the 0.5 threshold and it is not fooled by class imbalance, so it gives a
fuller picture than plain accuracy."

*解释：* AUC = 模型把随机一个 bottom 排在随机一个 top 之上的能力。0.5 = 瞎猜，1.0 = 完美。
它不依赖 0.5 阈值、不被类别不平衡骗，比单看准确率全面。

**Q33 — What is a confusion matrix?**

*答：* "It is a 2 by 2 table that shows the model's predictions against the true labels. It
tells me how many top and bottom campaigns were correct, and exactly what kind of mistakes the
model makes — whether it confuses top as bottom, or bottom as top."

*解释：* 混淆矩阵 = 2×2 表，预测 vs 真实。能看出模型错在哪个方向。

**Q34 — How do you turn a probability into a label? Why 0.5?**

*答：* "The model outputs a probability that a campaign is bottom-funnel. If it is above 0.5, I
label it bottom; otherwise top. 0.5 is the natural default. I also kept the probability itself
as a confidence score — predictions near 0.5 are uncertain and flagged for review."

*解释：* 模型输出"是 bottom 的概率"，>0.5 判 bottom，否则 top。0.5 是自然默认值。
我还保留了概率本身做可信度——靠近 0.5 的不确定，标记出来人工复核。

### E. 验证与反思

**Q35 — How do you know the model works on campaigns Stage A could not label?**

*答：* "Honestly, I cannot measure it directly, because I have no ground truth for those
campaigns. But I have three indirect signals. One — cross-validation held out whole customers,
so the AUC estimates how well it generalizes. Two — the model agrees with Stage A on 97.6% of
labeled campaigns, so it did not drift away from the rules. Three — I flag low-confidence
predictions for manual review. The proper fix is a hand-labeled test set."

*解释：* 老实说没法直接测，因为那些广告没标准答案。但有 3 个间接证据：
(1) 交叉验证留出整客户，AUC 估泛化；(2) 模型和 Stage A 一致率 97.6%，没跑偏；
(3) 低可信预测标记出来人工复核。彻底的解法是人工测试集。
**这题很重要——它就是作业明确问的问题。**

**Q36 — What is the model's biggest weakness?**

*答：* "There is no independent ground truth. The model is trained on labels I generated with
rules, and evaluated against the same kind of labels. So if the rules have a systematic bias,
the model learns that bias and my evaluation won't catch it. That is why a separate
hand-labeled test set is the most important next step."

*解释：* 最大弱点 = 没有独立标准答案。模型用规则标签训练，又拿同类标签评估。
规则若有系统性偏差，模型学到偏差，评估还发现不了。所以独立人工测试集是头号待办。

**Q37 — The model agrees with Stage A 97.6%. Is that good or bad?**

*答：* "It is mostly good — it shows the model is consistent with the rules and did not learn
something random. But 100% agreement would actually be bad — it would mean the model just
memorized the rules and adds nothing. The 2.4% disagreement is interesting: those are mostly
low-confidence Stage A labels, and the model may be correcting rule mistakes there."

*解释：* 基本是好事——说明模型和规则一致、没乱学。但 100% 一致反而不好，那等于模型只是背规则、没价值。
那 2.4% 不一致很有意思——大多是低可信标签，模型可能在那里纠正规则的错误。

**Q38 — How would you properly evaluate this without ground truth?**

*答：* "I would hand-label a random sample of maybe 50 campaigns myself, very carefully, before
looking at any predictions. That becomes a small but honest test set. Then I compare both the
rules and the model against it. It is small, but it is independent, which is what matters."

*解释：* 我会在看任何预测前，亲手仔细标 50 个随机广告做小测试集。然后规则和模型都拿它来比。
样本小但独立——独立才是关键。

**Q39 — What could go wrong in production?**

*答：* "New platforms or new objective wordings the rules never saw. Clients changing how they
name campaigns. The model would still output a label but with no warning that it is guessing.
That is why I output a confidence score — so low-confidence predictions can be caught and
reviewed instead of trusted blindly."

*解释：* 上线后可能：新平台、新目标写法、客户改命名习惯。模型照样输出标签但不会报警。
所以我输出可信度分数——低可信的能被抓出来复核，而不是盲信。

**Q40 — What is the difference between training score and test score here?**

*答：* "Training score is how well the model does on data it already saw — it is always
optimistic. Test score, here from cross-validation, is on data it did not see during training.
I always report the cross-validation score, because that is what predicts real-world
performance."

*解释：* 训练分数 = 在见过的数据上的表现，总是偏乐观。测试分数（这里来自交叉验证）= 没见过的数据。
我只汇报交叉验证分数，因为那才能预示真实表现。

### F. Part 2 — 流量相关性

**Q41 — Explain Part 2.**

*答：* "Part 2 uses the campaign labels from Part 1 to study website traffic. The question is:
if a customer runs mostly top-funnel or mostly bottom-funnel campaigns, does that show up in
how visitors arrive at their website? I measured each customer's funnel mix, joined it with
the traffic data, and looked at the correlations."

*解释：* Part 2 用 Part 1 的标签研究网站流量。问题：客户偏 top 还是偏 bottom，
能不能从访客怎么来网站看出来。算 funnel mix → join 流量 → 看相关性。

**Q42 — How did you measure "funnel mix"? Why spend share?**

*答：* "I used the share of spend that went to bottom-funnel campaigns — bottom spend divided
by total spend. I chose share, not raw spend, because raw spend just measures customer size; a
big spender would dominate. I chose spend, not campaign count, because spend reflects where
the customer actually puts money — a 5000-dollar campaign and a 10-dollar campaign should not
count the same."

*解释：* 用 bottom-funnel 花费占比（bottom 花费 ÷ 总花费）。
用占比不用绝对花费——绝对花费只反映客户大小，大客户会主导。
用花费不用广告数——花费才反映客户真金白银投在哪，5000 块和 10 块的广告不该算一样。
**这题作业明确要求 justify，一定背熟。**

**Q43 — How did you handle months with no campaigns?**

*答：* "The traffic table has a row for every customer and month, but some customer-months had
no campaigns at all. There the funnel mix is undefined. I dropped those rows — about 3.9% of
the data. I did not fill them with 0 or 0.5, because that would invent a strategy that did not
exist. I reported the count so it is transparent."

*解释：* 流量表每个客户每月都有行，但有些月份没投广告，funnel mix 算不出来。
我丢掉这些行（3.9%）。不填 0 或 0.5——那等于编造一个不存在的策略。我把数量报出来保持透明。

**Q44 — What did you find? Does funnel strategy show up in traffic?**

*答：* "Yes, very clearly. Customers with more bottom-funnel spend have much higher on-site
conversion rate, more direct traffic, and more branded search. Customers with more top-funnel
spend have more new visitors, higher bounce rate, more social traffic, and more non-branded
search. The correlations were strong — many above 0.7 in size."

*解释：* 是的，非常清晰。偏 bottom 的客户：转化率高、直接访问多、品牌词搜索多。
偏 top 的客户：新访客多、跳出率高、社交流量多、非品牌搜索多。相关性很强，很多绝对值 >0.7。

**Q45 — Correlation is not causation. What does that mean here?**

*答：* "It means I found that funnel mix and traffic patterns move together, but I cannot say
one causes the other. A bottom-funnel strategy might cause more branded search — or a strong
brand might cause both. My analysis shows the relationship exists; it does not prove the
direction."

*解释：* 意思是我发现 funnel mix 和流量模式一起变，但不能说谁导致谁。
偏 bottom 可能导致更多品牌词搜索——也可能是品牌本来就强，同时导致了两者。
分析证明关系存在，不证明方向。

**Q46 — What is Pearson correlation?**

*答：* "It is a number from -1 to +1 that measures how strongly two variables move together in
a straight line. +1 means they rise together perfectly, -1 means one rises as the other falls,
and 0 means no linear relationship."

*解释：* 皮尔逊相关系数，-1 到 +1，衡量两个变量"直线式"一起变的强度。
+1 完全同涨，-1 一涨一跌，0 没有线性关系。

**Q47 — The campaign labels are model predictions. Does that affect Part 2?**

*答：* "Yes, it does. Any error from Part 1 carries into Part 2. If the model mislabeled some
campaigns, the funnel mix is slightly wrong, and the correlations are slightly off. As a
robustness check, I could repeat Part 2 using only campaigns the model was very confident
about, and see if the conclusions still hold."

*解释：* 会。Part 1 的误差会传到 Part 2。模型标错广告 → funnel mix 略有偏 → 相关性略偏。
稳健性检查：只用模型很有把握的广告重做 Part 2，看结论还成不成立。

**Q48 — How would you explain this to a marketing manager?**

*答：* "I would skip the statistics and show the segmentation chart. I grouped customers into
three types — top-heavy, balanced, and bottom-heavy — and showed their typical traffic side by
side. A manager can see at a glance that bottom-heavy customers get more direct and branded
traffic and convert better. One clear picture beats a correlation table."

*解释：* 不讲统计，给他看分组柱状图。把客户分 top-heavy/balanced/bottom-heavy 三组并排对比。
经理一眼就懂。一张清楚的图胜过一张相关性表。

### G. 机器学习通用概念

**Q49 — What is overfitting?**

*答：* "Overfitting is when a model memorizes the training data, including its noise, instead
of learning the real pattern. It scores very well on training data but poorly on new data. I
guard against it with a simple model, regularization, and cross-validation."

*解释：* 过拟合 = 模型背下了训练数据连噪声一起背，没学到真规律。训练集分高，新数据上差。
我用简单模型、正则化、交叉验证来防。

**Q50 — What is the difference between supervised and unsupervised learning?**

*答：* "Supervised learning uses labeled examples — the model learns to predict a known answer.
Unsupervised learning has no labels — it finds structure on its own, like grouping similar
items. My project is supervised, but the twist is I had to create the labels first with rules.
That in-between approach is called weak supervision."

*解释：* 监督学习有标签，学着预测已知答案；无监督没标签，自己找结构（如聚类）。
我的项目是监督学习，但特别在标签是我自己用规则造的。这种中间地带叫弱监督。

**Q51 — What is precision and recall?**

*答：* "Take bottom-funnel as the target. Precision asks: of all campaigns I labeled bottom,
how many were really bottom? Recall asks: of all the real bottom campaigns, how many did I
catch? Precision is about not making false alarms; recall is about not missing things."

*解释：* 以 bottom 为目标。精确率：我判成 bottom 的里面，真是 bottom 的占多少（别误报）。
召回率：所有真 bottom 里，我抓到了多少（别漏）。

---

## 第 7 部分：答不上来时的救场话术

面试官就是想看你**不会的时候怎么反应**。慌乱、瞎编最糟。下面话术按场景分，挑两三句背熟。

### 场景 1：需要时间思考（先说这句，别冷场）

- *"That's a good question. Let me think for a moment."*
  （这是个好问题，让我想一下。）
- *"Let me take a second to organize my thoughts."*
  （让我花点时间整理一下思路。）

> 用法：任何难题先说这个，争取 5-10 秒。说的时候放慢，显得从容。

### 场景 2：没听懂问题（一定要问清，别瞎答）

- *"Could you rephrase that? I want to make sure I understand the question."*
  （能换个说法吗？我想确认我理解对了。）
- *"When you say X, do you mean ... ?"*
  （你说的 X，是指……吗？）

> 用法：英语不好，听不懂很正常。问清楚比答错强一百倍。

### 场景 3：完全不会（诚实 + 展示学习意愿）

- *"I haven't worked with that directly, so I don't want to guess. But I would be very keen to
  learn it."*
  （这个我没直接做过，我不想瞎猜。但我很愿意学。）
- *"I'm not familiar with that yet. Could you tell me a bit about it? I learn fast."*
  （这个我还不熟。你能稍微讲讲吗？我学得快。）

> 用法：彻底不会就**诚实承认**。面试官最反感不懂装懂。承认 + 表达学习意愿，反而加分。

### 场景 4：会一点，但没把握（说出思路，展示解决问题的能力）

- *"I'm not 100% sure, but my thinking would be ... Does that sound right?"*
  （我不是百分百确定，但我的思路是……这样对吗？）
- *"I didn't do that in this project, but if I did, I would start by ..."*
  （这个我项目里没做，但如果要做，我会先从……开始。）
- *"Let me reason through it out loud."*
  （我把思路说出来一起想。）

> 用法：这是最值钱的话术。面试官想看的是**思考过程**，不是标准答案。
> 边说边想，说错了也没关系。

### 场景 5：被追问"为什么这么做"，一时答不上

- *"My main reason was ... Looking back, an alternative could have been ..."*
  （我主要的理由是……回头看，另一个做法可能是……）
- *"That's a fair point. I'd reconsider that part."*
  （你说得有道理，那部分我会重新考虑。）

> 用法：被质疑别死扛。先给你当时的理由，再大方承认有别的做法。
> 这显示你**想得清楚、也听得进意见**。

### 场景 6：答错了被指出

- *"You're right, thank you for the correction. So the correct way is ..."*
  （你说得对，谢谢指正。那么正确的做法是……）

> 用法：错了就大方认。复述一遍正确答案，证明你当场学会了。

### 万能兜底句（什么都想不起来时）

- *"I want to give you an honest answer rather than guess. Right now I'm not sure, but I know
  how I'd find out — I would ..."*
  （我想给你一个诚实的回答而不是瞎猜。现在我不确定，但我知道怎么去搞清楚——我会……）

> 这句话几乎万能：诚实 + 展示你知道"怎么找答案"，这本身就是数据科学家最重要的能力。

### 心态提醒（中文，给自己看）

1. **英语不好不丢人**，说慢一点、用简单词、用短句。面试官在乎的是你的思路，不是语法。
2. **不会的题不要慌**。一场面试有一两题答不上来非常正常，没人全会。
3. **诚实 > 表演**。装懂一旦被戳穿，整场印象就垮了；坦诚反而显得可靠。
4. **多用"I would..."句式**。把"我不知道"变成"我会这样去解决"，立刻从被动变主动。
5. 实在卡住，回到你**最熟的部分**：把话题引回这个项目的两步法（Stage A / Stage B），那是你的主场。
