# Task 2 — Architecture Design: ขยาย Enterprise Knowledge Agent สู่ Production

**ขอบเขต:** พนักงาน 5,000 คน · 50,000 queries/วัน · เอกสารภายในผสมไทย-อังกฤษ
**สมมติฐาน:** peak ~20% ของปริมาณต่อวันอยู่ในชั่วโมงเร่งด่วน ≈ 3 QPS เฉลี่ย / ~10 QPS burst · corpus ~100k–200k เอกสาร ≈ 1–2M chunks
**เป้าหมาย SLO:** p95 ≤ 3 วินาที · availability 99.5% · hallucination rate < 2% · false-refusal < 5%

> **ข้อสังเกตสำคัญ:** 50,000 queries/วัน ไม่ใช่ปริมาณที่ใหญ่ (≈3 QPS) ต้นทุนหลักคือ **token ของ LLM** ไม่ใช่ infrastructure การออกแบบจึงเน้นลด token ก่อนเน้น scale-out

---

## 1. Cost vs. Latency

| ชั้น | เลือกใช้ | เหตุผล |
|---|---|---|
| **Embedding** | **BGE-M3 self-hosted** (1 GPU T4/L4) fallback เป็น `text-embedding-3-large` | เอกสารเป็นภาษาไทยผสมอังกฤษ โมเดล multilingual ให้ recall ดีกว่าอย่างมีนัยสำคัญ · self-host = ข้อมูลไม่ออกนอกองค์กร (ข้อบังคับสถาบันการเงิน) · ต้นทุนคงที่ ~$150/เดือน ไม่ผันตามปริมาณ · embed 1 query ≈ 20–40 ms |
| **Vector DB** | **Qdrant** (self-hosted, HNSW, 3 nodes) | ที่ 1–2M chunks ยังเล็กสำหรับ Qdrant · เหตุผลชี้ขาดคือ **payload filtering แบบ native** ซึ่งจำเป็นต่อ RBAC ในข้อ 3 (pre-filter ก่อน search) · pgvector ใช้ได้ถ้ามี Postgres อยู่แล้วแต่ filter + HNSW ยังด้อยกว่า · Pinecone/managed = ข้อมูลออกนอก ไม่ผ่าน governance |
| **LLM** | **สองชั้น:** ค่าเริ่มต้น `gpt-4o-mini` / Claude Haiku (~85% ของ traffic) → escalate ไปโมเดลใหญ่เมื่อ query ซับซ้อนหรือ confidence ก้ำกึ่ง | คำถามส่วนใหญ่คือ lookup ตรงไปตรงมา เมื่อ context ดี โมเดลเล็กตอบได้เท่ากัน · ส่วนต่างราคาโมเดลเล็ก/ใหญ่ ~10–20 เท่า จึงเป็นตัวแปรต้นทุนอันดับหนึ่ง |

**ประมาณการต้นทุน token** (context 4 chunks ≈ 2,000 input tokens, 300 output tokens/query)

| กรณี | Token/วัน | ต้นทุน/เดือน (ประมาณ) |
|---|---|---|
| ใช้โมเดลใหญ่ทั้งหมด | 100M in / 15M out | ~$10,000 |
| โมเดลเล็กทั้งหมด | 100M in / 15M out | ~$720 |
| **โมเดลเล็ก + escalate 15% + caching** | ~55M in | **~$450–600** |

### Caching Strategy (ลด token ~45–55%)

1. **Exact-match cache** (Redis, key = hash ของ query ที่ normalize แล้ว + ACL scope) — คำถาม support ภายในซ้ำกันสูงมาก คาด hit rate 30–40% · latency กรณี hit ≈ 50 ms
2. **Semantic cache** — ถ้า embedding ของ query ใกล้ query เดิม ≥ 0.97 ให้ใช้คำตอบเดิม เพิ่ม hit อีก ~10% · **ต้องแยก cache ตาม ACL scope** มิฉะนั้นจะกลายเป็นช่องรั่วของสิทธิ์
3. **Provider prompt caching** — system prompt + few-shot คงที่ ~500 tokens ทุก request ใช้ prompt caching ลดราคาส่วนนี้ ~90%
4. **Embedding cache แบบ content-hash** — เอกสารที่ไม่เปลี่ยน ไม่ re-embed ตอน re-index (ประหยัดรอบ ingestion เป็นหลัก)
5. **TTL 24 ชม. + invalidate ทันทีเมื่อเอกสารต้นทางเปลี่ยน** — ป้องกันการตอบด้วยนโยบายเก่า ซึ่งอันตรายกว่าการตอบช้า

**Latency budget (p95, cache miss):** guardrail 5 ms → embed 40 ms → vector search 30 ms → LLM TTFT ~700 ms → streaming จนจบ ~2.0 s → validation 50 ms ≈ **2.8 s** · ตอบแบบ streaming ทำให้ผู้ใช้เห็นตัวอักษรแรกภายใน ~0.8 s ซึ่งสำคัญต่อ perceived latency มากกว่าเวลารวม

---

## 2. Evaluation Framework

**Golden set:** 200–300 คู่คำถาม-คำตอบต่อ domain (HR / Finance / IT / Procurement) กำหนดและเซ็นรับรองโดย SME ของแต่ละ BU ทบทวนทุกไตรมาส แยกชุด `in-scope`, `out-of-scope`, `injection`, `no-evidence` เหมือนที่ prototype ทำแล้วแต่ขยายขนาด

| มิติที่วัด | วิธีวัด | เป้าหมาย |
|---|---|---|
| **Retrieval Recall@5** | chunk ที่ถูกต้องอยู่ใน top-5 หรือไม่ (label ระดับ chunk) | ≥ 0.90 |
| **Retrieval Precision@5 / MRR** | สัดส่วน chunk ที่เกี่ยวข้องจริง · อันดับของ chunk ที่ถูก | Precision ≥ 0.6 |
| **Answer Accuracy** | LLM-as-judge เทียบกับ golden answer ตาม rubric + สุ่มตรวจโดยคน | ≥ 90% |
| **Hallucination Rate** | ทุกประโยคที่เป็นข้อเท็จจริงต้อง entail ได้จาก chunk ที่อ้างอิง (NLI + judge) | **< 2%** |
| **False-refusal Rate** | คำถามที่ตอบได้แต่ระบบปฏิเสธ | < 5% |
| **False-answer Rate** | คำถามนอกขอบเขต/ไม่มีหลักฐาน แต่ระบบตอบ | < 1% |

**หลักการที่ต้องระวัง:** LLM-as-judge ต้องถูก calibrate กับ label ของมนุษย์ก่อนใช้ (ต้องการ agreement ≥ 0.85) มิฉะนั้นเรากำลังวัดด้วยไม้บรรทัดที่ไม่รู้ความยาว

**Online metrics:** thumbs up/down ต่อคำตอบ · อัตราการ escalate ไปหาคน · อัตราการคลิกดู source · การกระจายของ confidence score (ใช้ตรวจ corpus drift) · อัตรา fallback แยกตามเหตุผล

**Regression gate:** ทุกการเปลี่ยน prompt / โมเดล / chunking / threshold ต้องรัน golden set ใน CI ถ้าคะแนนตกเกิน 2% บล็อกการ deploy — ป้องกันกรณีคลาสสิกที่ "แก้ prompt ให้คำถามหนึ่งดีขึ้น แล้วอีกสิบคำถามพัง"

**Human review:** สุ่ม 1% ของ traffic จริงให้ SME ตรวจรายสัปดาห์ กรณีที่ตรวจแล้วผิด ให้ไหลกลับเข้า golden set อัตโนมัติ

---

## 3. Security & Governance (RBAC)

**หลักการเดียวที่ห้ามผ่อน: filter ก่อน retrieve ไม่ใช่หลัง generate**
การ retrieve ทั้งหมดแล้วค่อยกรองคำตอบทีหลังถือว่าล้มเหลวแล้ว เพราะเนื้อหาลับได้เข้าไปอยู่ใน prompt ของ LLM ไปแล้ว และยังรั่วผ่านอันดับผลลัพธ์กับเวลาตอบสนองได้

| ชั้น | การควบคุม |
|---|---|
| **Ingestion** | ทุก chunk พก payload `classification` (public / internal / confidential / restricted) และ `acl_groups` ที่ **สืบทอดจากสิทธิ์ของระบบต้นทาง** (SharePoint / File Server / HR system) · sync สิทธิ์ทุกคืน · เอกสารที่หา ACL ไม่ได้ → deny by default |
| **Query** | resolve กลุ่มของผู้ใช้จาก SSO token (Entra ID / AD) → ส่งเป็น Qdrant filter `acl_groups ∈ user_groups AND classification ≤ user_clearance` → HNSW ค้นเฉพาะ subset ที่มีสิทธิ์ |
| **ข้อมูลระดับผู้บริหาร** (เช่น เงินเดือน) | จัดเป็น `restricted` + แยก collection ต่างหาก + ต้อง step-up authentication · PII ที่ไม่จำเป็นต่อการตอบให้ redact ตั้งแต่ ingestion ไม่ใช่ตอน generate |
| **Cache** | key ต้องผูกกับ ACL scope เสมอ — cache ที่ไม่แยกสิทธิ์คือช่องรั่วที่พบบ่อยที่สุดในระบบลักษณะนี้ |
| **Audit** | log `user_id`, กลุ่มที่ resolve ได้, `doc_ids` ที่ถูกดึง, เวลา — **ไม่ log เนื้อหาเอกสารและไม่ log ความลับ** · เก็บแบบ immutable ตามรอบ compliance |
| **Untrusted content** | เนื้อหาในเอกสารถือเป็น input ที่ไม่น่าเชื่อถือ ห้ามให้ข้อความในเอกสารสั่งงานระบบหรือเรียก tool ได้ · ป้องกัน indirect prompt injection |
| **Secrets** | เก็บใน Vault / Secrets Manager เท่านั้น ไม่มี key ใน repo หรือใน log |

**Data residency:** ชั้น `confidential` ขึ้นไป ให้ประมวลผลด้วยโมเดลที่ deploy ใน VPC/on-prem เท่านั้น ชั้น `internal` จึงจะใช้ LLM API ภายนอกได้ — ข้อนี้กระทบการเลือกโมเดลในข้อ 1 จึงต้องตัดสินร่วมกับ Compliance ตั้งแต่ต้น ไม่ใช่ตอน go-live

---

## 4. Code Structure, Observability & Operations

### โครงสร้างโค้ด

```
api/          FastAPI, auth, rate limit          ← เปลี่ยนบ่อย, logic น้อย
pipeline/     guardrails, retrieval, generation, validation
              ← ฟังก์ชันบริสุทธิ์ ตัดสินใจล้วน ๆ ทดสอบได้โดยไม่ต้องต่อ network
adapters/     llm, vectorstore, docstore, cache  ← interface บาง ๆ สลับผู้ให้บริการได้
config/       ค่า threshold, ชื่อโมเดล, เวอร์ชัน prompt ทั้งหมดอยู่ที่เดียว
evals/        golden set + runner ที่ใช้ทั้งใน CI และ ad-hoc
```

เหตุผล: ชั้น `pipeline` คือส่วนที่ตัดสินใจเรื่องความปลอดภัยทั้งหมด จึงต้องไม่มี I/O ปนอยู่ เพื่อให้เขียนเทสต์ครอบทุกเส้นทางได้โดยไม่ต้องมี API key (prototype ทำแบบนี้แล้วด้วย 50 เทสต์)

### Tracing & Monitoring

- **OpenTelemetry** span ต่อขั้นตอน (`guardrail` → `embed` → `search` → `generate` → `validate`) แนบ attribute: คะแนน retrieval, doc_ids, เวอร์ชัน prompt, ชื่อโมเดล, จำนวน token, cache hit/miss · `trace_id` เดียวกับ `request_id` ที่ผู้ใช้เห็นบนหน้าจอ
- **Structured JSON logs** → Loki/OpenSearch (ต่อยอดจาก `logs/rag.jsonl` ของ prototype โดยตรง)
- **Prometheus + Grafana:** p50/p95/p99 แยกตาม stage · fallback rate แยกตามเหตุผล · refusal rate · การกระจาย confidence · token spend ต่อวัน · cache hit rate · error rate
- **Alert:** fallback rate > 15% (ส่วนใหญ่แปลว่า index เพี้ยนหรือเอกสารหาย) · p95 > 5s · error rate > 1% · ค่าใช้จ่ายเกิน budget รายวัน · การกระจาย retrieval score เลื่อน (สัญญาณ corpus drift)

### RCA & Incident Management

- **Replayability:** จาก `request_id` เดียว ต้องย้อนได้ว่า retrieve อะไรมา คะแนนเท่าไร ใช้ prompt เวอร์ชันไหน โมเดลอะไร index snapshot ไหน และ gate ไหนเป็นคนตัดสิน — ทำได้เพราะทุกค่าเหล่านี้ถูกปักเวอร์ชันและบันทึกไว้ ไม่ใช่ค่าที่ลอยตามเวลา
- **Kill switch:** feature flag บังคับให้ระบบตอบ fallback อย่างเดียว หรือปิด generation ชั่วคราวโดยยังค้นหาได้ — ใช้เมื่อพบ hallucination เป็นระบบ
- **Rollback:** prompt / โมเดล / index เป็น artifact ที่มีเวอร์ชัน ย้อนกลับได้แยกจากกัน
- **Severity ladder:** S1 = ตอบข้อมูลผิดชั้นความลับ หรือระบบล่ม · S2 = accuracy ตกเกิน threshold · S3 = latency เสื่อม · S4 = คำถามเดี่ยวตอบผิด (เข้าคิว golden set ไม่ใช่ incident)

### Handover สู่ D2 Operation

1. **Runbook ต่อ alert** — ทุก alert ต้องมีเอกสารว่า "เห็นแบบนี้ ให้ตรวจอะไร แก้อย่างไร escalate เมื่อไร" alert ที่ไม่มี runbook ให้ถือว่ายังไม่พร้อมส่งมอบ
2. **Dashboard เดียวสำหรับ on-call** — สุขภาพระบบ + คุณภาพคำตอบในหน้าจอเดียว
3. **Change process:** แก้ prompt / threshold ต้องผ่าน PR + eval gate เท่านั้น ห้ามแก้บน production โดยตรง
4. **เจ้าของ golden set:** ระบุชื่อ SME ต่อ domain ที่รับผิดชอบทบทวนรายไตรมาส — ถ้าไม่มีเจ้าของ ชุดวัดจะเก่าและตัวเลขจะโกหกภายในหกเดือน
5. **Known-issues log + ข้อจำกัดที่ยอมรับแล้ว** ส่งมอบพร้อมโค้ด เพื่อให้ทีมรับช่วงแยกออกว่าอะไรคือบั๊ก อะไรคือขอบเขตที่ตกลงกันไว้

---

**สรุปการตัดสินใจหลัก:** ปริมาณงานระดับนี้ไม่ต้องการสถาปัตยกรรมที่ซับซ้อน สิ่งที่ต้องลงแรงคือ (1) เลือกโมเดลให้เหมาะกับภาษาไทยและข้อบังคับด้านข้อมูล (2) ลด token ด้วย cache และการเลือกโมเดลเป็นชั้น (3) บังคับสิทธิ์ที่ชั้น retrieval ไม่ใช่ชั้นคำตอบ (4) วัดผลด้วย golden set ที่มีเจ้าของจริง เพราะสิ่งที่วัดไม่ได้ก็ปรับปรุงไม่ได้และส่งมอบไม่ได้
