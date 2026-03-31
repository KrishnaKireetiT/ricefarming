# Pipeline Testing Protocol
## Extension Officer Evaluation of Pipeline A vs Pipeline B

**Document:** 1 Million Hectare Project Handbook (Mekong Delta Rice Farming)
**Pipelines Under Test:**
- **Pipeline A** — anonymised
- **Pipeline B** — anonymised

**Evaluators:** Extension officers familiar with Mekong Delta rice farming practices
**App URL:** http://35.186.40.29:8443

---

## 1. Testing Overview

| Parameter | Value |
|---|---|
| Topics | 10 |
| Questions per topic | 5 (free-form, your own questions) |
| **Total questions per evaluator** | **50** |
| Pipelines tested per question | 2 (both run automatically) |
| Total pipeline runs | 100 (50 × 2) |

**Key point:** You choose your own questions. The topics below guide *what* to ask about, but the exact wording is entirely up to you. Ask questions the way a real farmer or field officer would.

---

## 2. Topics

For each topic, think of **5 questions** you would naturally ask about the subject. Mix different types of questions: simple factual lookups, step-by-step procedures, calculations, and comparisons. You may ask in **English or Vietnamese** (or a mix).

| # | Topic | Handbook Reference | What to ask about |
|---|---|---|---|
| 1 | **Land Preparation** | Ch. 2.1 | Laser leveling, soil prep, drainage ditches, field conditions before sowing |
| 2 | **Seed Preparation & Sowing** | Ch. 2.2–2.3 | Seed treatment, soaking, sowing methods (row, cluster, broadcast), machinery specs |
| 3 | **Water Management / AWD** | Ch. 2.4 | Irrigation scheduling, AWD technique, monitoring tubes, water depth thresholds |
| 4 | **Fertilization** | Ch. 2.5 | Nutrient rates by soil type/season, timing, application methods, buried fertilizer |
| 5 | **Pest Management / IPM** | Ch. 2.6 | Spray thresholds, biological control, specific pests (BPH, leaf folder, snails, rats) |
| 6 | **Harvest & Post-Harvest** | Ch. 3 | Harvest timing, grain moisture, drying methods, storage, quality maintenance |
| 7 | **Straw Management** | Ch. 4.1–4.3 | Straw collection timing, stubble handling, burning alternatives, seasonal differences |
| 8 | **Straw Utilization** | Ch. 4.4 | Mushroom cultivation, composting, animal feed, mulching |
| 9 | **Unit Conversions & Calculations** | Cross-cutting | công ↔ hectare, dosage math, machinery capacity calculations, area-based rates |
| 10 | **Technology & Digital Innovation** | Ch. 5 | Drones, precision agriculture, digital tools, remote sensing |

---

## 3. How to Use the App

### Step 1: Login
1. Open the app at **http://35.186.40.29:8443**
2. Log in with your assigned credentials (or create an account)
3. The app defaults to **Extension Officer** mode — no need to change anything

### Step 2: Ask a Question (Evaluate Tab)
1. Pick a topic from the list above
2. Type your question in the text box (in English or Vietnamese)
3. Click **"Run Both Pipelines"**
4. Both Pipeline A and Pipeline B will process your question simultaneously
5. You will see both answers side by side

### Step 3: Score Both Answers
For each question, you must fill in **all** of the following:

#### a) Ground Truth Answer
Write what **you** consider the correct, complete answer to the question. This is the gold standard — what an expert would say.

#### b) Score Each Pipeline (1–5 scale)

| Criterion | What to evaluate | Scale |
|---|---|---|
| **Factual Accuracy** | Are the facts, numbers, and recommendations correct? | 1 (wrong) → 5 (perfect) |
| **Completeness** | Does it cover all relevant aspects? | 1 (missing most) → 5 (covers everything) |
| **Relevance** | Does it actually answer the question asked? | 1 (off-topic) → 5 (directly addresses it) |
| **Specificity** | Does it give concrete details (dosages, timing, measurements)? | 1 (vague) → 5 (precise) |
| **Language & Clarity** | Is it clear, well-structured, and farmer-friendly? | 1 (confusing) → 5 (crystal clear) |
| **Safety (no harm)** | Could following this advice damage crops or harm farmers? | 1 (dangerous) → 5 (completely safe) |

#### c) Failure Mode Tags
If you notice problems in either answer, tag them:
- Wrong facts
- Missing info
- Wrong language
- Off-topic
- Hallucination (invented info)
- Too vague
- Outdated info
- Harmful advice

#### d) Overall Preference
Choose one:
- **Pipeline A is better**
- **Pipeline B is better**
- **Tie (both equal)**

#### e) Notes (optional)
Any additional observations, specific errors you noticed, or suggestions.

### Step 4: Submit & Repeat
1. Click **"Submit Evaluation"**
2. Move to your next question for the same topic, or switch to a new topic
3. Repeat until you have **5 questions per topic × 10 topics = 50 questions total**

### Step 5: Review Your Results (Results Tab)
- The **Results** tab shows your evaluation summary: average scores, preference counts, failure mode distribution
- You can **edit** any previous evaluation by clicking the ✏️ button
- **Export** your results as CSV, JSON, or Excel

---

## 4. Scoring Guide

| Score | Meaning |
|---|---|
| **5** | Excellent — fully correct, complete, specific details, proper citations |
| **4** | Good — mostly correct, minor omissions |
| **3** | Adequate — partially correct but missing important details |
| **2** | Poor — significant errors or very incomplete |
| **1** | Fail — wrong answer, irrelevant, or no useful information |

---

## 5. Tips for Good Questions

- **Be specific:** "How much nitrogen for alluvial soil in Winter-Spring?" is better than "Tell me about fertilizer"
- **Mix difficulty:** Include simple fact questions, multi-step procedures, and calculation questions
- **Use local terms:** Try công, sạ hàng, AWD, etc. to test how well the system handles them
- **Try Vietnamese:** At least 1–2 questions per topic in Vietnamese
- **Ask what farmers ask:** Think about real questions you get from farmers in the field

---

## 6. Tracking Sheet

Use this to track your progress. Check off each topic as you complete 5 questions.

| # | Topic | Q1 | Q2 | Q3 | Q4 | Q5 | Done? |
|---|---|---|---|---|---|---|---|
| 1 | Land Preparation | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 2 | Seed Preparation & Sowing | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 3 | Water Management / AWD | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 4 | Fertilization | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 5 | Pest Management / IPM | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 6 | Harvest & Post-Harvest | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 7 | Straw Management | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 8 | Straw Utilization | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 9 | Unit Conversions & Calculations | ☐ | ☐ | ☐ | ☐ | ☐ | |
| 10 | Technology & Digital Innovation | ☐ | ☐ | ☐ | ☐ | ☐ | |

---

## 7. Deliverables

| What | How |
|---|---|
| 50 evaluated questions (scores + ground truth) | Submitted via the app |
| Exported results | Download from Results tab → Excel/CSV |
| Overall feedback | Notes field in each evaluation |
