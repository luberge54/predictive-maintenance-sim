# Step 4 — The dashboard

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

The report in `python -m src.decision` already prints every number. The dashboard exists
for the one thing a printed report cannot do: **let the reader move the assumption and
watch the answer move.** An assumption you can drag is an assumption you can argue with.

---

## 1. What is on the page

| Section | Question it answers |
|---|---|
| Sidebar sliders | What do *you* think a failure costs, and how much notice does your shop need? |
| Thresholds | What those two answers imply — including the cheapest threshold, shown so its uselessness is visible |
| What each policy would have cost | The backtest: three policies priced on the same 20 engines |
| How much your assumption matters | The full curve from 1x to 20x, including where this tool is not worth using |
| The fleet right now | 100 live engines, most urgent first, with a recommendation each |
| One engine, in detail | Its predicted life against both thresholds, its raw sensors, and the instruction in plain words |

---

## 2. Two fleets, deliberately not the same engines

**The backtest** uses evaluation engines from the training file. They ran to failure, so a
policy can be replayed on them and priced.

**The live fleet** is the 100 NASA test engines, whose recording stops before they fail —
exactly the situation a plant is in: still flying, outcome unknown. A policy cannot be
*scored* on them, but a recommendation can be *made* for them.

The live table shows an **Actual RUL** column. That is known only because this is a
benchmark; the model never sees it and neither would an operator. It is there so a reader
can check the recommendations against reality instead of taking them on trust.

---

## 3. Replay once, price many times

Moving a slider used to mean replaying 131 fleets: about 1.4 seconds of dead air per drag,
and drawing the 39-point assumption curve would have taken nearly a minute.

The fix came from noticing that **what happens to a fleet does not depend on the cost
model — only the price of what happened does.** Overhauls, failures and scrapped cycles at
each candidate threshold are computed once and cached; changing the cost ratio re-prices a
stored table with three multiplications.

| | Before | After |
|---|---|---|
| Slider response | 1.4 s | **0.27 s** |
| Full assumption sweep | ~55 s | under 1 s |

The two paths are kept honest by a test asserting that pricing a row of the grid gives
exactly the same `FleetOutcome` as replaying that threshold directly.

---

## 4. What the page refuses to hide

- **The unusable optimum is displayed**, next to the usable one, labelled
  "optimal and unusable". A tool that quietly applied the operational constraint without
  showing what it cost would be hiding its most interesting decision.
- **The saving curve runs to 1.0x**, where the answer is −8.1% and a calendar is the better
  policy. Truncating the axis at the flattering part would be the easiest lie on the page.
- **The threshold curve is flat**, and the caption says why rather than hoping nobody
  notices: every engine's prediction eventually drops below 5 cycles, so a low threshold
  already catches the fleet and the cost ratio has nothing left to buy.
- **No trained model produces instructions, not a traceback** — the exact commands to run,
  with the dataset URL.

---

## 5. Tested, not just eyeballed

`tests/test_app.py` renders the real page in-process with Streamlit's `AppTest` and asserts
that it raises nothing, that all five sections appear, that a higher failure cost produces
a higher saving, that demanding more notice moves the threshold, that the fleet table is
ordered most-urgent-first, and that a missing model is handled as a setup step.

They skip rather than fail when the dataset or model is absent, so a fresh clone still
runs a green suite.

---

## 6. Regenerating the screenshots

The two images in the README are captured from the running app, so they cannot drift from
what the code actually renders:

```powershell
pip install playwright          # dev-only; deliberately not in requirements.txt
streamlit run app.py --server.headless true --server.port 8532
```

Then drive a browser at it, hide Streamlit's own toolbar and deploy button — they are not
part of this project — and shoot at `device_scale_factor=2` so the text stays crisp.
Chrome or Edge already installed on the machine works via `channel="chrome"`, which avoids
downloading a second browser.

---

## 7. Limitations

- **Streamlit reruns the whole script on every interaction.** Correct here because
  everything expensive is cached, but it is the reason the caching is deliberate rather
  than incidental.
- **The live fleet is a snapshot.** Each engine is shown at its last recorded cycle. There
  is no notion of time passing, no new readings arriving, no history of past decisions.
- **One reader at a time.** No accounts, no saved settings, no audit trail of who was told
  what. A real deployment would need all three.
