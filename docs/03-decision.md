# Step 3 — From prediction to decision

Reproduce every number on this page with:

```powershell
.\.venv\Scripts\python.exe -m src.decision
```

This is the part of the project with an opinion. The model says *"about 23 cycles left"*.
A plant manager needs *"pull it now"* or *"it can wait"*. Bridging those two is not a
modelling problem — it is a cost problem, and the cost assumption is stated out loud
rather than buried in a threshold someone picked.

---

## 1. The cost model, in numbers

Everything is priced in units of **one planned overhaul**, so the currency never matters —
only the ratios do.

| | Cost | Where it comes from |
|---|---|---|
| Planned overhaul | 1.0 | the unit |
| Unplanned failure | 5.0 | the assumed cost ratio — **an operator input, exposed as a slider** |
| One wasted cycle of good life | 0.00495 | derived: `1 / 202`, so scrapping a full engine life costs exactly one extra overhaul |

```
total_cost = overhauls x 1.0  +  failures x 5.0  +  wasted_cycles x 0.00495
```

That single number prices every policy below. Nothing else is compared.

---

## 2. The threshold is found, not chosen

Every candidate threshold from 1 to 150 cycles is replayed across the fleet — *intervene
the first cycle the model predicts this much life or less* — and priced. The cheapest
wins. Move the cost ratio and the threshold moves on its own.

Here is that search on the 20 tuning engines:

| Threshold | Overhauls | Failures | Wasted cycles | Total cost |
|---|---|---|---|---|
| 1 | 0 | 20 | 0 | 100.00 |
| 3 | 18 | 2 | 39 | 28.19 |
| **5** | 20 | 0 | 113 | **20.56** |
| 8 | 20 | 0 | 167 | 20.83 |
| 20 | 20 | 0 | 382 | 21.89 |
| 50 | 20 | 0 | 954 | 24.72 |
| 150 | 20 | 0 | 3,832 | 38.97 |

**The cost curve is a cliff followed by a gentle slope.** Below a threshold of about 5,
engines start slipping through and failing — each one costs five overhauls. Above it,
every engine is caught and raising the threshold only throws away good life, which is
cheap. So the optimum lands on the cliff edge.

---

## 3. The finding: the cost-optimal policy is unusable

The unconstrained answer is **"intervene when the model predicts 5 cycles of life left"**.

That is arithmetically correct and operationally worthless. No maintenance organisation
sources a part, books a hangar slot and rosters a crew inside five flights. A
recommendation with five cycles of notice is not a decision, it is a warning that you are
about to have an unplanned failure with extra paperwork.

The cost model from step 0 has a hole: **it prices wasted life and unplanned failure, but
puts no value on notice.** Left uncorrected, the tool confidently recommends something a
domain expert would reject in one sentence.

The fix is not to fudge the arithmetic. It is to add the missing operational input:

```python
MINIMUM_LEAD_TIME = 20   # cycles of notice the shop needs before it can act
```

Like the cost ratio, this is an **operator input, not a fact about the data**. There is no
correct value in the dataset. It belongs to whoever runs the fleet, and it is a candidate
for a second slider.

### The constraint made the policy better, not worse

The expected trade-off was giving up some cost to gain usability. That is not what
happened. Priced on the 20 evaluation engines:

| Threshold | Overhauls | Failures | Wasted | Cost |
|---|---|---|---|---|
| 5 — the unconstrained optimum | 19 | **1** | 73 | 24.36 |
| 20 — floored at the lead time | 20 | **0** | 344 | **21.70** |

The knife-edge optimum was **overfitted to the tuning fleet**. It sat exactly at the point
where those particular 20 engines were all caught, and on 20 different engines one slipped
through and failed. The operationally realistic threshold has margin, and margin
generalises.

A constraint that comes from the domain rather than the data made the answer both more
usable *and* cheaper on engines nobody had seen.

---

## 4. Results

Thresholds derived on the tuning engines, at a 5.0x cost ratio:

| | Cycles | |
|---|---|---|
| Cheapest threshold, ignoring lead time | 5 | optimal and unusable |
| **Action threshold** | **20** | floored at the lead time |
| **Watch threshold** | **33** | action + 13, the model's own typical error |
| Calendar interval (baseline) | 137 | tuned by the same search |

Priced on the 20 evaluation engines — which shaped neither the model nor the threshold,
and were read exactly once:

| Policy | Overhauls | Failures | Wasted cycles | Total cost |
|---|---|---|---|---|
| Run to failure | 0 | 20 | 0 | 100.0 |
| Calendar, every 137 cycles | 19 | 1 | 1,133 | 29.6 |
| **Condition-based (this project)** | **20** | **0** | **344** | **21.7** |

### Headline

> **26.7% cheaper than a cost-optimised calendar policy. 78.3% cheaper than running to
> failure.**

The calendar baseline was given the same optimisation pass as our own policy — beating a
deliberately badly tuned baseline would prove nothing. Even at its best interval it still
loses one engine outright *and* scraps 1,133 cycles of good life, because a fixed date
cannot tell a strong engine from a weak one. Condition-based monitoring loses none and
scraps 344.

---

## 5. What an operator actually reads

Every recommendation is generated from rules. Same engine, same sentence, every run, with
a line of code behind each clause.

**Engine 1, cycle 1**
> No action. Predicted remaining life is 131 cycles, 98 clear of the point where the
> model's uncertainty starts to matter. At the assumed 5.0x cost ratio the action
> threshold sits at 20 cycles. Re-check in 48 cycles.

**Engine 1, cycle 162**
> Monitor closely. Predicted remaining life is 32 cycles, above the 20-cycle action
> threshold that the assumed 5.0x cost ratio sets, but by less than the model's typical
> error of 13 cycles. The model calls this engine safe and is not confident enough to be
> believed. Re-check every cycle; do not schedule work yet.

**Engine 1, cycle 175**
> Intervene now. Predicted remaining life is 19 cycles, at or below the 20-cycle action
> threshold. That threshold is where an unplanned failure costing 5.0x a planned overhaul
> makes pulling this engine cheaper on average than running it further. Typical model
> error is plus or minus 13 cycles, so treat this as a range, not a date.

Each one states the recommendation, the prediction, **the cost assumption it rests on**,
and the model's uncertainty. A reader who disagrees with the 5.0x assumption knows exactly
which number to argue with.

### Why rules and not a language model

A maintenance instruction has to be reproducible and auditable. The same inputs must
produce the same sentence on every run, and an engineer must be able to point at the line
of code that produced each clause. Text that is rephrased on every call cannot be
diffed, cannot be tested, and cannot be defended in an incident review. It also means the
whole project runs offline, with no API key and no per-run cost.

### The "monitor closely" band

It is not a hand-picked buffer. It is exactly **one typical model error wide** — the zone
where the model says an engine is safe, and is routinely wrong by enough that it might not
be. Showing that band is more honest than reporting a single number and hoping.

---

## 6. Limitations

- **20 evaluation engines is a small sample.** 26.7% is one draw from one seeded split.
  Repeated splits would give a range, and the honest reading is "roughly a quarter".
- **The cost ratio and the lead time are assumptions, not measurements.** Both are stated
  and both are adjustable. That is the design, but it means the headline number is
  conditional on someone else's numbers being close to these.
- **One intervention per engine.** Each engine is pulled at most once and the trajectory
  ends. No renewal, no second life, no fleet-level capacity limit — the policy would
  happily recommend grounding every engine the same week.
- **Perfect execution is assumed.** "Intervene now" is taken to happen immediately and to
  always succeed.
- **The evaluation engines all run to failure.** The official NASA test engines are cut
  off before failing, so a policy cannot be replayed on them: there is no way to know what
  would have happened after the recording stops. This is why the evaluation set is carved
  out of the training file.
