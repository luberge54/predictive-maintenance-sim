# Step 5 — The narrative

What this project argues, and how to defend it out loud.

---

## 1. The pitch

**One sentence.**

> I took a public engine-degradation dataset, built a deliberately ordinary failure model,
> and spent the real effort on the layer above it: turning a prediction into a maintenance
> decision under an explicit, adjustable cost trade-off.

**Ninety seconds.**

> Most portfolio ML projects stop at "here is my RMSE". That answers a question nobody in a
> business asks. The question a plant actually asks is: *given what the model predicts,
> what should we do, and what does it save us?*
>
> So the model is boring on purpose — gradient boosting on rolling sensor features,
> library defaults, one seed. It reaches an RMSE of 13.2 on the standard NASA protocol,
> which is competitive with published classical results and about a cycle worse than deep
> learning. That gap changes no decision, and I'll explain why in a moment.
>
> The interesting part is the decision layer. It prices three policies against each other
> under one explicit cost model, replays every candidate threshold across the fleet, and
> keeps the cheapest. Against a calendar policy that was given the same optimisation, it
> comes out **26.7% cheaper**, with zero failures against the calendar's one — measured on
> engines that shaped neither the model nor the threshold.
>
> And the two things I got wrong on the way are the parts I'd actually want to talk about.

---

## 2. The trade-off, and why it is the whole project

Two ways a maintenance call can be wrong, and they are not symmetric:

- **Too early** — the engine is pulled while still healthy. You pay for an overhaul you did
  not need yet, and you scrap the life it had left.
- **Too late** — the engine fails in service. You pay for an unplanned repair, downtime,
  and whatever it took with it.

Everything is priced in units of **one planned overhaul**, so the currency never matters:

```
total_cost = overhauls x 1.0  +  failures x RATIO  +  wasted_cycles x (1 / 202)
```

Three deliberate choices sit in that line.

**The failure cost is a slider, not a constant.** It is an assumption about someone's
business, and no dataset contains it. Hard-coding it would be pretending to know something
I don't. The dashboard plots the whole range from 1x to 20x, including the end where the
answer is "keep your calendar".

**The wasted-cycle cost is derived, not chosen.** Scrapping a full nominal engine life
costs exactly one extra overhaul, so the rate falls out as `1 / median_life`. Without this
term the cheapest policy is "replace every engine on cycle 1" — it never fails, so it never
pays the failure cost. That absurdity is the reason the term exists.

**The threshold is found, not picked.** Every candidate from 1 to 150 cycles is replayed
across the fleet and priced. Nobody chose 20.

---

## 3. The two things I got wrong

These are the answers worth leading with. A project with no wrong turns is a project where
nothing was actually run.

### The cost-optimal policy was unusable

The optimiser's first answer was **"intervene when the model predicts 5 cycles of life
left"**. Arithmetically correct. Operationally worthless — no shop sources a part, books a
hangar slot and rosters a crew inside five flights.

The cost model had a hole: it priced wasted life and unplanned failure, but put no value on
**notice**. I added the missing operator input rather than fudging the arithmetic.

The result was not the expected trade of cost for usability:

| Threshold | Failures | Cost on unseen engines |
|---|---|---|
| 5 — cost-optimal | 1 | 24.36 |
| 20 — realistic notice | **0** | **21.70** |

The knife-edge optimum sat exactly where those particular tuning engines were all caught,
and on twenty different engines one slipped through. **A constraint from the domain rather
than the data made the answer both usable and cheaper.** Margin generalises.

### The premise about the slider was false

I had written, in three places, "move the cost ratio and the threshold moves with it". Then
I ran the sweep.

The threshold stops responding above a ratio of 1.5x. Every engine's prediction eventually
drops below 5 cycles, so a threshold of 5 already catches the entire fleet; raising it
prevents no further failure however expensive failure becomes. The cost ratio has nothing
left to buy.

The honest reframe is better than the original claim:

> **The cost ratio does not decide *when* to intervene. It decides *whether this tool is
> worth using at all*.** At 1.0x it is 8.1% *worse* than a calendar. It starts paying above
> roughly 1.5x, and reaches 51.3% at 20x.

The dashboard shows the losing end of that curve. A tool that cannot tell a reader "keep
your calendar" is a sales pitch, not an analysis.

---

## 4. Questions I expect, and honest answers

**"The model is gradient boosting with default settings. Why should I be impressed?"**

You shouldn't be, by the model. It is ordinary on purpose. The hours that would have gone
into hyperparameter search went into the cost model, the three-way split, and the policy
replay instead — because that is where the decision quality actually lives. I did make one
model choice on evidence: gradient boosting over random forest, decided on error *near
failure* (7.85 vs 9.57) rather than headline RMSE, because being eight cycles wrong about
an engine with 300 cycles left changes nothing.

**"An LSTM would beat you on this dataset."**

Yes — published deep-learning results land around 12 to 13 RMSE against my 13.2. That is
roughly one cycle of accuracy. It would change **zero** decisions here, because the
intervention threshold is set by how much notice the shop needs, not by how precise the
model is. Spending weeks to buy a cycle of precision that no policy consumes is the exact
mistake this project is arguing against.

**"26.7%, on twenty engines. That is noise."**

Fair. It is one draw from one seeded split, and the honest reading is "roughly a quarter,
on this fleet, under these assumptions". The direction is robust — condition-based beats a
calendar at every ratio above 1.5x — but the second decimal place is not. With more time I
would run repeated splits and report a range rather than a point.

**"Your cost ratio is made up."**

It is, and so is the lead time. Both are stated in the scoping document, both are sliders,
and the dashboard plots the whole range so a reader can find their own number on the curve.
That is the design, not an oversight: the alternative is burying an invented constant in
the code where nobody can argue with it.

**"It's simulated data."**

It is. Real fleet telemetry is dirtier — dropouts, drift, recalibrations — and the
degradation signal here is cleaner than anything in service. What survives the move to real
data is the decision layer, which does not care where the prediction came from. What breaks
first is the assumption that a prediction is available every cycle for every unit.

**"How would you deploy this?"**

I wouldn't, yet. It is missing everything operational: fleet-level capacity limits (it will
happily recommend grounding forty engines the same week), parts lead times beyond the
single notice parameter, any notion of a decision being recorded and later reviewed, and
monitoring for the model drifting away from the fleet it was fitted on.

---

## 5. What the engineering discipline was for

Three findings in this project would have been invisible without it, and each one was
caught by a test or a check rather than by luck:

| What was wrong | How it surfaced | What it would have cost |
|---|---|---|
| Dead sensors detected with `std > 0` | Sensor count changed between runs | Two constant sensors have a std near `1e-15`; the model silently got 15, 17 or 18 columns depending on the split |
| Threshold tuned and scored on the same engines | Noticed while designing step 3 | The headline saving would have been inflated and unfalsifiable |
| Model pickled as a custom class | Loading it from a second entry point | `__main__` gets baked into the artefact; the dashboard would have failed |

Plus two silent-cheat paths closed before they could flatter the results: rolling windows
that read the future, and a row-wise split putting cycle 99 in training and cycle 100 in
validation. Both have dedicated tests, because both produce a model that scores beautifully
and fails in service.

71 tests. No network access, no API key, no per-run cost.

---

## 6. What I would do next

In rough order of how much each would change the answer:

1. **Repeated splits.** Replace the single seeded draw with several, and report the saving
   as a range. This is the weakest part of the current result.
2. **Fleet-level capacity.** Cap how many engines can be pulled in a window, and let the
   policy sequence them. Today's answer is per-engine and ignores the hangar.
3. **A prediction interval, not a point.** The "monitor closely" band uses one global RMSE.
   A per-engine uncertainty would let the band tighten where the model is confident.
4. **The harder subsets.** FD002 and FD004 have six operating conditions, which forces
   per-condition normalisation and would test whether the decision layer holds up when the
   model is genuinely worse.
5. **Cost sensitivity as an output.** Report, per engine, how far the cost ratio would have
   to move before its recommendation flips. That turns the slider from a demo into a
   robustness statement.

---

## 7. The sentence to remember

> I did not build a model that predicts failure. I built a tool that decides what to do
> about it — and it can tell you when not to bother.
