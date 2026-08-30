# System Design

ระบบนี้ออกแบบสำหรับพนักงาน 5,000 คน ปริมาณประมาณ 50,000 คำถามต่อวัน และเอกสารภาษาไทย/อังกฤษ

**เป้าหมายหลัก:** เริ่มแสดงคำตอบภายใน 3 วินาที, uptime 99.5%, hallucination ต่ำกว่า 2% และ false-refusal ต่ำกว่า 5%

50,000 คำถามต่อวันคิดเป็นเฉลี่ยประมาณ 3 QPS ซึ่งไม่ต้องใช้โครงสร้างพื้นฐานซับซ้อน ต้นทุนหลักมาจาก **token ของ LLM ไม่ใช่ค่า server** จึงควรลดจำนวน token ก่อนขยายเครื่อง

## 1. Cost vs. Latency

| ส่วน | ทางเลือกที่เลือก | เหตุผลสำคัญ |

| Embedding | BGE-M3 แบบ self-host | รองรับไทย/อังกฤษในเอกสารเดียวกัน และกรณีถามไทยแต่เอกสารเป็นอังกฤษ ถ้าใช้โมเดลภาษาเดียว recall จะลดลง · ประมวลผลเองทำให้ข้อมูลไม่ออกนอกองค์กร และค่า GPU คงที่ |
| Vector DB | Qdrant | กรองสิทธิ์ (`payload filter`) ไปพร้อมกับ vector search จึงค้นเฉพาะเอกสารที่ผู้ใช้มีสิทธิ์ตั้งแต่ต้น (ดูข้อ 3) |
| LLM | Gemini 2.5 Flash-Lite | ราคา $0.10 / $0.40 ต่อ 1M input/output tokens คุณภาพใกล้ GPT-4o mini ที่ $0.15 / $0.60 สำหรับงาน RAG ลักษณะนี้ |

Pinecone ไม่เหมาะเพราะข้อมูลต้องไม่ออกนอกองค์กร ส่วน pgvector มีข้อจำกัดมากกว่าเมื่อใช้ filter ร่วมกับ HNSW ในขนาดข้อมูลที่คาดไว้

### การลดต้นทุนด้วย cache

- **exact-match cache** ใน Redis สำหรับคำถามที่ซ้ำกัน คาดว่า hit ได้ประมาณ 40%
- **semantic cache** เมื่อ embedding ของคำถามใหม่ใกล้คำถามเดิมตั้งแต่ 0.97 ขึ้นไป
- เก็บ embedding ตาม `content hash` เอกสารที่ไม่เปลี่ยนจึงไม่ต้องคำนวณใหม่เมื่อ re-index
- TTL 24 ชั่วโมง และล้าง cache ทันทีเมื่อเอกสารต้นทางเปลี่ยน เพื่อไม่ให้ตอบด้วยนโยบายเก่า
- **ต้องผูก cache key กับ ACL scope เสมอ** มิฉะนั้นผู้ใช้อาจได้คำตอบจากข้อมูลที่ตนไม่มีสิทธิ์อ่าน

## 2. Evaluation Framework

สร้าง **golden set** ประมาณ 200–300 ข้อต่อหน่วยงาน (HR, Finance, IT, Procurement) ให้ SME รับรอง แบ่งเป็น in-scope, out-of-scope, prompt injection และ no-evidence เพื่อวัดทั้ง "ตอบถูก" และ "รู้ว่าเมื่อไรไม่ควรตอบ"

| ตัวชี้วัด | วิธีวัด | เป้าหมาย |
|---|---|---:|
| Retrieval Recall@5 | chunk ที่มีคำตอบอยู่ใน top-5 หรือไม่ | ≥ 90% |
| Retrieval Precision@5 | สัดส่วน chunk ที่เกี่ยวข้องจริงใน top-5 | ≥ 60% |
| Answer Accuracy | เทียบ golden answer ด้วย LLM-as-judge และสุ่มตรวจโดยคน | ≥ 90% |
| Hallucination rate | ข้อเท็จจริงทุกประโยคต้องอ้างกลับไปยัง chunk ได้ | < 2% |
| False-refusal rate | คำถามที่มีคำตอบ แต่ระบบปฏิเสธ | < 5% |
| ตอบทั้งที่ไม่มีหลักฐาน | คำถามนอกขอบเขต แต่ระบบยังตอบ | < 1% |

ต้องวัด retrieval แยกจากคุณภาพคำตอบ เพราะคำตอบผิดอาจเกิดจาก "ค้นไม่เจอ" หรือ "ค้นเจอแล้วสรุปผิด" ซึ่งแก้คนละวิธี หากใช้ LLM-as-judge ต้องเทียบกับผลตรวจของคนให้ตรงกันอย่างน้อย 85% ก่อนใช้จริง

**Regression gate:** ทุกครั้งที่เปลี่ยน model, prompt, chunking หรือ threshold ต้องรัน golden set หากคะแนนลดลงเกิน 2% ให้บล็อกการ deploy

**การวัดใน production:** เก็บ feedback, การเปิดดู source และ fallback rate พร้อมสุ่มคำตอบจริง 1% ให้ SME ตรวจทุกสัปดาห์ คำตอบที่พบว่าผิดจะนำกลับเข้า golden set

## 3. Security & Governance

หลักการสำคัญคือ **กรองสิทธิ์ก่อนค้น ไม่ใช่ค้นทั้งหมดแล้วค่อยกรองคำตอบ** เพราะหากเอกสารลับถูกส่งเข้า LLM ก็ถือว่ารั่วแล้ว แม้ไม่ปรากฏในคำตอบ ผู้ใช้ยังอนุมานได้จากอันดับผลลัพธ์หรือเวลาที่ระบบใช้ตอบ

- ทุก chunk ต้องมี `classification` (`public / internal / confidential / restricted`) และ `acl_groups` ที่สืบทอดจากระบบต้นทาง (SharePoint, File Server, HR) และ sync ทุกคืน ระบบนี้ไม่ควรกำหนดสิทธิ์ชุดใหม่เอง
- ตอนรับคำถาม resolve กลุ่มผู้ใช้จาก SSO แล้วส่งเป็น Qdrant filter: `acl_groups ∩ user.groups ≠ ∅ AND classification ≤ user.clearance`
- ใช้ filter เดียวกันตรวจเงื่อนไขอื่นได้ เช่น `region` และวันเริ่มใช้/หมดอายุของเอกสาร เพื่อไม่ให้คนละสาขาเห็นข้อมูลกัน หรือหยิบนโยบายเก่ามาตอบ
- ข้อมูลระดับผู้บริหาร เช่น เงินเดือน จัดเป็น `restricted` แยก collection และใช้ step-up authentication
- หากตรวจสอบ ACL ไม่ได้ ให้ถือว่าไม่มีสิทธิ์ (deny by default)
- ลบ PII ที่ไม่จำเป็นตั้งแต่ ingestion · ถือว่าเนื้อหาในเอกสารไม่น่าเชื่อถือ ห้ามใช้เป็นคำสั่งให้ระบบเรียก tool · เก็บ API key ใน secrets manager
- audit log ที่แก้ไขไม่ได้ มี `user_id`, กลุ่มที่ใช้ตัดสิน, `doc_ids` และเวลา แต่ไม่เก็บเนื้อหาเอกสาร

**Governance** — ระบบบังคับใช้สิทธิ์ได้ตามค่าที่ตั้งไว้ แต่ต้องมีคนรับรองว่าค่านั้นถูกต้อง

- **Data owner ของแต่ละ BU** กำหนด `classification` และ ACL ทบทวนทุกไตรมาส และอนุมัติเอกสารเข้า/ออก index — เอกสารที่ไม่มีเจ้าของห้ามเข้า index
- **SME** ดูแล golden set และสุ่มตรวจคำตอบ · **ทีมพัฒนา** เปลี่ยน prompt/model/threshold ผ่าน PR + eval gate เท่านั้น
- **Compliance + Legal** อนุมัติการส่งข้อมูลออก API ภายนอก ข้อมูลระดับ confidential ขึ้นไปต้องประมวลผลใน VPC เท่านั้น ประเด็นนี้ต้องตกลงตั้งแต่ต้นเพราะมีผลต่อการเลือกโมเดลในข้อ 1

## 4. Code structure, Monitoring และ Operations

```text
api/       รับ request, ตรวจผู้ใช้ และ rate limit
pipeline/  guardrail, retrieval, generation และ validation
adapters/  เชื่อมต่อ LLM, Vector DB และ cache
config/    threshold, ชื่อโมเดล และ prompt version
evals/     golden set และ runner สำหรับ CI/ad-hoc
```

`pipeline/` รวมการตัดสินใจเรื่องสิทธิ์และการปฏิเสธคำตอบไว้ที่เดียว และไม่ควรมี network I/O เพื่อให้ทดสอบทุกเส้นทางได้โดยไม่ต้องใช้ API key ส่วนการเปลี่ยนผู้ให้บริการทำที่ `adapters/` โดยไม่กระทบ logic หลัก

**Tracing และ Monitoring:** ใช้ OpenTelemetry สร้าง span ต่อขั้นตอน (`guardrail → embed → search → generate → validate`) แนบ retrieval score, `doc_ids`, prompt version, model, จำนวน token และ cache hit/miss โดยใช้ `trace_id` เดียวกับ `request_id` ที่ผู้ใช้เห็น ส่ง structured log ไป Loki/OpenSearch และใช้ Grafana ดู p50/p95/p99 แยกตาม stage รวมถึง fallback rate, refusal rate, token spend และ cache hit rate

**แจ้งเตือนเมื่อ:** p95 เกิน 5 วินาที, error rate เกิน 1%, fallback rate เกิน 15%, ค่าใช้จ่ายเกินงบรายวัน หรือ retrieval score เปลี่ยนการกระจายอย่างมีนัยสำคัญ (สัญญาณ corpus drift)

**RCA และ Incident management:** เป้าหมายคือ replay ได้จาก `request_id` เดียวว่าระบบดึงเอกสารอะไรมา คะแนนเท่าไร ใช้ prompt/model/index เวอร์ชันใด และ fallback เพราะเหตุใด จึงต้องปักเวอร์ชันของ prompt, threshold และ index ไว้เสมอ ต้องมี **kill switch** ที่สั่งให้ตอบ fallback อย่างเดียวได้ทันที และ rollback prompt/model/index แยกจากกันได้ · ระดับความรุนแรง: **S1** ข้อมูลผิดชั้นความลับหรือระบบล่ม · **S2** accuracy ต่ำกว่าเป้า · **S3** ตอบช้า · **S4** คำถามเดี่ยวตอบผิด (บันทึกเข้า golden set ไม่ถือเป็น incident)

**การส่งมอบให้ D2 Operation:** ทุก alert ต้องมี runbook ระบุว่า "เห็นอาการนี้ต้องตรวจอะไร แก้อย่างไร escalate เมื่อไร" — alert ที่ไม่มี runbook ถือว่ายังส่งมอบไม่ได้ · มี dashboard เดียวที่เห็นทั้งสุขภาพระบบและคุณภาพคำตอบ · ห้ามแก้ prompt หรือ threshold บน production โดยตรง · ระบุเจ้าของ golden set ของแต่ละหน่วยงาน เพราะชุดทดสอบที่ไม่มีเจ้าของจะล้าสมัยจนตัวเลขไม่น่าเชื่อถือ · ส่งมอบ known issues พร้อมโค้ด
