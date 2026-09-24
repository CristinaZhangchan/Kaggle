# 🎤 Speaking Script — Campaign Funnel Classification
# 8-10 分钟 | 面向非技术听众 | 简单英文口语

> **使用方法**：每一页 PPT 对应一段话。`[pause]` 表示停顿一拍。
> 全部用短句、简单词。说慢、说清楚。不用背 PPT 上的字，说下面的话。

---

## 📊 时间分配

| Slide | 内容 | 时间 | 累计 |
|:---:|---|:---:|:---:|
| 1 | Title | 0:30 | 0:30 |
| 2 | The Problem | 1:15 | 1:45 |
| 3 | Two Steps | 1:00 | 2:45 |
| 4 | Five Clues | 1:30 | 4:15 |
| 5 | Step 1 Results | 1:00 | 5:15 |
| 6 | The Model | 1:00 | 6:15 |
| 7 | Model Results | 0:45 | 7:00 |
| 8 | Honest Limitations | 1:00 | 8:00 |
| 9 | Traffic Findings | 1:15 | 9:15 |
| 10 | Conclusions | 0:45 | 10:00 |

---

## Slide 1 — Title（30 秒）

> Hi everyone. Thank you for your time today. `[pause]`
>
> My project is called **Campaign Funnel Classification**. `[pause]`
>
> In simple words — I built a system that **automatically labels online ads** based on their purpose. And then I used those labels to discover something interesting about website traffic. `[pause]`
>
> Let me walk you through it.

---

## Slide 2 — The Problem（1 分 15 秒）

> Imagine you work at an advertising company. You help your clients run ads on Google, Facebook, TikTok, and LinkedIn. `[pause]`
>
> Every ad has a **purpose**. `[pause]`
>
> Some ads are designed to reach **new people** — people who have never heard of the brand. We call this **top-funnel**. Think of it as **"saying hello to strangers."** `[pause]`
>
> Other ads are designed to **sell to people who already know the brand** — people who visited the website before, or added something to their cart. We call this **bottom-funnel**. Think of it as **"closing the deal."** `[pause]`
>
> **The problem is** — in our data, there is **no label** that tells us which type each ad is. Nobody marked it. `[pause]`
>
> And this matters. Because you **cannot** judge a "hello" ad by the same standard as a "selling" ad. They need **different budgets, different targets, and different reports**. `[pause]`
>
> So my task was: **give every ad a label — top or bottom — automatically.**

---

## Slide 3 — Two Steps（1 分钟）

> Here is the challenge. `[pause]`
>
> A machine learning model is like a **student**. It needs **examples to learn from**. But we have **zero examples** — nobody labeled anything. `[pause]`
>
> So I used a **two-step** approach. `[pause]`
>
> **Step one** — I used **simple rules** to label the ads I was confident about. For example: *"If the ad targets people who already visited the website, it is probably a selling ad."* These rules gave me a **first batch of labels**. `[pause]`
>
> **Step two** — I used those labels to **teach a model**. The model learned the patterns, and then it labeled **all 637 ads** — including the ones the rules couldn't figure out. `[pause]`
>
> This approach is called **weak supervision** — you start with imperfect labels, then let a model improve from there.

---

## Slide 4 — Five Clues（1 分 30 秒）

> Let me show you how Step one works. I found **five clues** in the data. Each clue votes — top or bottom. `[pause]`
>
> **Clue one — Who is the ad targeting?**
> If it targets "remarketing" — meaning people who already visited the website — that's bottom-funnel, a selling ad. If it targets broad, general audiences — that's top-funnel, a hello ad. `[pause]`
>
> **Clue two — What is the campaign's goal?**
> If the goal says "awareness" or "reach" — top. If it says "conversions" or "sales" — bottom. `[pause]`
>
> **Clue three — What does the ad look like?**
> A video is usually for storytelling — top. A form asking for your email or a product catalog — that's bottom. `[pause]`
>
> And there are **two more similar clues** — what the ad is optimizing for, and keywords in the campaign name. `[pause]`
>
> Each clue votes. If **most clues agree**, I assign that label. And I track **how confident** I am — based on how many clues agreed. Three or more clues agreeing = high confidence.

---

## Slide 5 — Step 1 Results（1 分钟）

> So how well did the rules work? `[pause]`
>
> They labeled **92 percent** of all campaigns. `[pause]`
>
> For **53 percent**, confidence was **high** — three or more clues all agreed. Another 32 percent had a clear majority. `[pause]`
>
> Only **8 percent** couldn't be labeled at all — no clue was strong enough. `[pause]`
>
> You can see on the chart that when two different clues both had an opinion, they **agreed most of the time** — which means the clues are reliable and checking each other. `[pause]`
>
> A key decision: I only used the **high and medium confidence** labels to train the model. The noisy ones were kept out — so the model only learns from good examples.

---

## Slide 6 — The Model（1 分钟）

> Now, Step two — training the model. `[pause]`
>
> I took those **540 campaigns** with confident labels and trained a classifier. `[pause]`
>
> I tried two models — a simple one and a more complex one — to see if the extra complexity was worth it. `[pause]`
>
> One important thing I did: when I tested the model, I made sure that **all ads from the same client** were on the same side — either all in training, or all in testing. `[pause]`
>
> Why? Because one client names all their ads in a **similar style**. If some are in training and some in testing, the model could just **recognize the style** instead of learning the real pattern. That would give a **fake high score**. `[pause]`
>
> By separating clients, I got an **honest** test result.

---

## Slide 7 — Model Results（45 秒）

> The results. `[pause]`
>
> The simple model actually performed **slightly better** — with a score of **0.994 out of 1.0**. So I chose the simpler one. When a simple approach works just as well, take the simple one. `[pause]`
>
> Final output: **321 ads labeled as top-funnel, 316 as bottom-funnel**. `[pause]`
>
> And the model agreed with the original rules on **97.6 percent** of cases — it learned the right patterns, and the small disagreements were mostly on the uncertain cases where the model may actually be correcting rule mistakes.

---

## Slide 8 — Honest Limitations（1 分钟）

> Now, I want to be honest about the limitations. `[pause]`
>
> The rules labeled 92 percent with measurable confidence. The model generalizes well — tested by holding out whole clients. The scores are strong. `[pause]`
>
> **But** — there is no independent "answer key." `[pause]`
>
> The model is trained on labels that **I** created with rules. And it's tested against the same kind of labels. So if my rules have a **blind spot** — the model inherits it, and the test won't catch it. `[pause]`
>
> The most important next step would be to **manually check about 50 random campaigns myself** — carefully, before looking at any predictions. That small set becomes a **truly independent test**. `[pause]`
>
> It's small, but it's **independent** — and that's what matters.

---

## Slide 9 — Traffic Findings（1 分 15 秒）

> Now, Part two — a follow-up question. `[pause]`
>
> Once we know each client's advertising strategy — mostly top-funnel or mostly bottom-funnel — **can we see a difference** in how people visit their website? `[pause]`
>
> I joined the campaign labels with a new dataset — each client's monthly website traffic data. `[pause]`
>
> The answer is **yes, very clearly**. `[pause]`
>
> Clients who spend more on **bottom-funnel ads** — the "closing the deal" type — their websites get more **direct visits**, more **branded search**, and much **higher conversion rates**. People come because they already know the brand. `[pause]`
>
> Clients who spend more on **top-funnel ads** — the "hello to strangers" type — their websites get more **new visitors**, more **social media traffic**, and **higher bounce rates**. People come, look around, but many leave. `[pause]`
>
> You can see it clearly in these charts. The advertising strategy leaves a visible **fingerprint** on the website traffic.

---

## Slide 10 — Conclusions（45 秒）

> To wrap up. `[pause]`
>
> **What I delivered**: A system that labeled all 637 ads — each with a label, a probability, and a confidence score. `[pause]`
>
> **What I found**: The type of advertising a company does clearly shows up in their website traffic patterns. `[pause]`
>
> **What I'd improve**: Manually label a small sample as a truly independent test — the most important next step. `[pause]`
>
> Thank you. I'm happy to take any questions.

---

## 💡 救场/应急话术

如果被问到不会的问题，用这些：

| 场景 | 说 |
|---|---|
| 需要时间想 | *"That's a good question. Let me think for a moment."* |
| 没听懂 | *"Could you rephrase that? I want to make sure I understand."* |
| 完全不会 | *"I haven't worked with that directly, so I don't want to guess. But I'd be keen to learn."* |
| 会一点 | *"I'm not 100% sure, but my thinking would be..."* |
| 被质疑 | *"That's a fair point. My main reason was... but looking back, an alternative could be..."* |
| 万能兜底 | *"I want to give an honest answer rather than guess. I'm not sure right now, but I know how I'd find out."* |

---

## 🔑 只需记住 3 个数字

| 数字 | 含义 |
|:---:|---|
| **92%** | 规则覆盖率（Step 1 标了 92% 的广告） |
| **0.994** | 模型准确度分数 (AUC)，满分 1.0 |
| **97.6%** | 模型和规则的一致率 |

---

## 🎯 练习建议

1. **第一遍**：对着脚本完整读一遍，计时（目标 9-10 分钟）
2. **第二遍**：半看半说，开始用自己的话
3. **第三遍**：只看 PPT，不看脚本，自由发挥
4. **语速**：宁可慢，不要快。每句话结尾停一拍。
5. **手势**：讲到图表时，手指着 PPT 说 "You can see here..."
6. **眼神**：看听众，不看屏幕。只有指图表时瞄一眼。
