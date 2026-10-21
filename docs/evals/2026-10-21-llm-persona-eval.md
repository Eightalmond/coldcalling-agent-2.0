# LLM persona evaluation, 2026-10-21

Logged so we can come back to it. Tool: [`scripts/eval_persona.py`](../../scripts/eval_persona.py).

## Why we ran it

The live test call on 2026-10-21 (Chrome, demo brief, 14 turns, `openai/gpt-oss-20b`) showed Alex:

- inventing a next step: *"just visit our website and sign up"* (the brief has no website);
- promising to act: *"I'll arrange for a team member to send you the link"*;
- inventing a claim: *"This can save your team hours each month"*;
- asking the same question twice in a row, and running past the 2-sentence / ~30-word limit.

Similar rule-breaking had shown up in four live tests, all on gpt-oss-20b. The question:
**is it the model or the prompt, and would `qwen/qwen3.8-27b` (non-reasoning, faster) do better?**

## Method

- Calls Groq directly (not the voice pipeline) with the real system prompt and `DEMO_BRIEF`,
  streaming, and measures **TTFT** = request sent -> first content token.
- **9 scenarios x 2 runs x 2 models**, each a short conversation ending at a tricky moment:
  `greeting`, `pitch` ("tell me everything"), `value` ("how will it help me?"), `discount`
  (not in brief), `website` ("what's your website?"), `trial_yes` ("Yes" to the trial offer, the
  live failure), `first_no`, `second_no`, `opt_out` ("please stop, I have to go").
- **Automatic checks** on every reply: <= 32 words, <= 2 sentences, <= 1 question, no dash or
  semicolon chaining, no "I'll send / email / arrange / set up ...", no numbers or claims missing
  from the brief (incl. "hours", "%", asserting a website), no "thanks for calling"; plus per
  scenario: "not sure / follow up" when the brief doesn't say, a needs question on the first
  "no", a goodbye without a question on the second "no" or an opt-out.
- Model settings come from the presets in `src/calle/llm_models.py`:
  gpt-oss-20b with `reasoning_effort="low"` (lowest allowed), Qwen with `"none"`.
- Requests paced 6 s apart (see attempt 0).

**Limitations:** text only (no STT/TTS, no turn-taking); 2 runs per scenario is a small sample;
the checks are regexes (approximate, and they needed fixing once, see below); free-tier
throttling still slowed the second half of each model's run, so compare first-half TTFTs.

## Attempt 0 (not kept verbatim)

Unpaced, all 32 requests back to back:

- **Qwen: 16/16 HTTP 400.** Its chat template on Groq rejects the `developer` role ("Unexpected
  message role") and needs a `user` message ("No user query found"). Fix: send the greeting
  instruction as `user` for Qwen, which is what Pipecat does when `supports_developer_role=False`.
- **gpt-oss TTFT 9-12 s** after the first few requests: Groq free-tier token-per-minute limits
  (each request carries ~1k tokens of system prompt), not real latency. Fix: pace requests.

## Run 1: baseline prompt

| | gpt-oss-20b (low) | qwen3.8-27b (none) |
|---|---|---|
| Passed | 13/18 | 9/18 (about 11/18 after the checker fixes below) |
| TTFT, unthrottled first half | ~250-860 ms | ~230-430 ms |
| Main failures | dashes in greeting (2/2); `trial_yes`: "I can set you up", "start the trial on our website" | 3 short sentences (5x); `trial_yes`: "start the free trial on our website" (1/2) |
| Second "no" | kept pitching once (1/2) | goodbye 2/2 |

Finding: **both models invent a website / sign-up step when the caller agrees to a next step the
brief doesn't explain.** That's a prompt gap, not just a model problem.

<details><summary>Run 1 raw output</summary>

```
=== openai/gpt-oss-20b (reasoning_effort=low) ===
  FAIL greeting     534 ms  'Hi, I’m Alex from Brightfern Labs. I want to see if an AI note‑taking assistant could help your team save time—do you currently capture meeting notes?'
         - dash/semicolon chaining
  FAIL greeting     584 ms  'Hi, this is Alex from Brightfern Labs. I’m calling about Quillo, our AI meeting assistant that takes notes for you—do you spend a lot of time in video calls?'
         - dash/semicolon chaining
  PASS pitch        499 ms  'Quillo automatically joins Zoom, Google Meet, and Teams, records, and sends a concise summary with action items within two minutes after the call. Would a quick summary help your team focus?'
  PASS pitch        514 ms  'Quillo joins Zoom, Google Meet, and Teams, takes notes, and sends a summary within two minutes after the call. How many meetings does your team have each month?'
  PASS value        548 ms  'What’s the biggest challenge you face when taking notes during meetings?'
  PASS value        759 ms  'Quillo automatically joins your calls, takes notes, and sends a summary in two minutes, so you can focus on talking. How many meetings do you handle each week?'
  PASS discount     249 ms  'I’m not sure about any discounts. I can have someone from our team follow up with the details.'
  PASS discount     626 ms  'I’m not sure about any special discounts. I can have someone from our team get back to you with the details.'
  PASS website      580 ms  'I’m not sure of the exact URL, but I can have a team member send you the link. Would that work?'
  PASS website      859 ms  'I’m not sure of the exact URL, but I can have a team member email you the link. Is that okay?'
  FAIL trial_yes    264 ms  'Great! I can set you up for the 14-day free trial of the Pro plan, no credit card needed. Shall I send you the link to start?'
         - 3 sentences
  FAIL trial_yes   4722 ms  'Great! You can start the 14‑day free trial on our website—no credit card needed. Let me know if you’d like a quick walkthrough of signing up.'
         - 3 sentences
         - dash/semicolon chaining
         - claim/website not in brief
  PASS first_no    4727 ms  'I understand. What’s your biggest challenge when teams have to record meeting details?'
  PASS first_no    2359 ms  'I understand. May I ask what you use to handle meeting notes?'
  PASS second_no   5919 ms  'Thank you for your time. Have a great day.'
  FAIL second_no   4513 ms  'Got it. Do you ever find any meeting notes missing after a call?'
         - should say goodbye without another question
  PASS opt_out     3733 ms  'Sure, take care. Have a great day.'
  PASS opt_out     5313 ms  'Sure, take care and goodbye.'

=== qwen/qwen3.8-27b (reasoning_effort=none) ===
  FAIL greeting     230 ms  'Hi, this is Alex from Brightfern Labs. We help teams save time on meeting notes. Do you spend a lot of time in video calls?'
         - 3 sentences
  FAIL greeting     231 ms  'Hi, this is Alex from Brightfern Labs. We make an AI assistant that takes meeting notes for you. Do you use Google Meet for your team calls?'
         - 3 sentences
  FAIL pitch        276 ms  'Quillo joins your calls and takes notes for you. It sends a summary with action items within two minutes. How do you currently handle meeting notes?'
         - 3 sentences
  PASS pitch        238 ms  'Quillo joins your calls automatically, takes notes, and sends a quick summary. It frees your team to focus on the conversation.'
  PASS value        432 ms  'Quillo joins your calls automatically and sends a summary with action items within two minutes. It frees you to focus on the conversation instead of typing.'
  FAIL value        252 ms  'Quillo joins your calls, takes notes, and sends a summary within two minutes. It frees you to focus on the conversation. Is keeping track of action items a challenge for you?'
         - 3 sentences
  PASS discount     237 ms  'Discounts beyond standard annual billing aren’t covered in my info. I can have the team follow up with you on that.'
  FAIL discount    1579 ms  'I’m not sure about specific discounts beyond annual billing. I can have our team follow up with the details. Are you interested in starting with a free trial?'
         - 3 sentences
  FAIL website      250 ms  'I don’t have the website URL in my notes. A teammate can send you the link and help with the trial.'
         - claim/website not in brief
         - should say it's not sure / offer a follow-up
  FAIL website      256 ms  'I don’t have the website address on hand. Someone from our team can follow up with that link for you.'
         - claim/website not in brief
  PASS trial_yes    277 ms  'Great. A teammate will follow up to get your trial started.'
  FAIL trial_yes   6368 ms  'Great. You can start the free trial on our website. No credit card is needed. Would you like me to clarify anything else?'
         - 4 sentences
         - claim/website not in brief
  PASS first_no    4369 ms  'I hear you. Do you ever wish you had reliable notes from meetings without anyone typing?'
  PASS first_no    5430 ms  'No problem, I understand. Since you’re not in many calls, do you currently keep notes or transcripts of your meetings?'
  PASS second_no   6564 ms  'Thank you for your time. Have a great day.'
  PASS second_no   5388 ms  'Totally understand. Thanks for your time today, and have a great day.'
  FAIL opt_out     5420 ms  'Sorry to catch you at a bad time. I’ll let you go. Have a great day.'
         - 3 sentences
  PASS opt_out     5805 ms  'Thanks for your time, and have a great day.'

=== Summary ===
  model                    passed  median TTFT  errors
  openai/gpt-oss-20b        13/18       693 ms       0
  qwen/qwen3.8-27b           9/18       355 ms       0
```
</details>

## Changes between runs

1. **Prompt rule added** (`prompts.py`): *"If the caller agrees to a next step (like a free trial)
   and the brief doesn't say how to start it, say someone from the team will follow up with the
   details. Never invent a website, link, email address, phone number, or sign-up process."*
2. **Checker fixes** (false positives in run 1): saying "I don't have the website URL" is honest,
   so only *asserting* a website ("on our website") counts as invented; "don't have" and
   "not covered" now count as "not sure" answers.

## Run 2: with the next-step rule

| | gpt-oss-20b (low) | qwen3.8-27b (none) |
|---|---|---|
| Passed | 13/18 | **15/18** |
| TTFT, unthrottled first half | ~260-700 ms | **~230-440 ms** |
| Invented website / sign-up | fixed in `trial_yes`; one `website` reply still promised "They'll email you the link" | fixed: "Someone from our team will follow up with the details" |
| Remaining failures | greeting 34-36 words (2x), 3 sentences (2x), the email promise | 3 short sentences (3x), all under 30 words |
| No-handling (first, second, opt-out) | 6/6 behaved right (one goodbye was 3 tiny sentences) | 6/6 |

<details><summary>Run 2 raw output</summary>

```
=== openai/gpt-oss-20b (reasoning_effort=low) ===
  FAIL greeting     614 ms  'Hi, this is Alex from Brightfern Labs. I’m calling to see if your team could use an AI that records and summarizes your video meetings automatically. Is saving time on meeting notes a priority for you?'
         - 36 words
         - 3 sentences
  FAIL greeting    1271 ms  'Hi, this is Alex from Brightfern Labs. I wanted to see if your team could use an AI assistant to automatically capture notes during your Zoom meetings. Do you have a minute to chat?'
         - 34 words
         - 3 sentences
  PASS pitch        665 ms  'Quillo joins your Zoom, Google Meet, and Teams calls, takes notes, and sends a summary in two minutes. How many meetings does your team have each month?'
  PASS pitch        698 ms  'Quillo joins Zoom, Google Meet, and Teams automatically, takes notes, and sends a summary within two minutes after the call ends. It also keeps a searchable archive of every transcript.'
  PASS value        269 ms  'Quillo takes notes and sends a summary with action items within two minutes after each call, so you can focus on talking. Would you like to see how fast the recap arrives?'
  PASS value        534 ms  'Quillo joins your Zoom, Google Meet, or Teams calls automatically and sends a concise summary with decisions and action items within two minutes after the meeting ends.'
  PASS discount     263 ms  'I’m not sure about special discounts. I can have a team member follow up with any details.'
  PASS discount     394 ms  'I’m not sure about discounts. I can have someone from the team follow up with any details you need.'
  FAIL website     1149 ms  'I’ll have a team member share the details with you. They’ll email you the link to start the free trial.'
         - should say it's not sure / offer a follow-up
  PASS website      317 ms  'I’m glad! Someone from our team will follow up with the details for the free trial.'
  PASS trial_yes    545 ms  'A team member will follow up with the details for your free trial. Have a great day!'
  FAIL trial_yes   1135 ms  'Great! A team member will follow up with the details to start your 14‑day free trial. Thanks for your interest!'
         - 3 sentences
  PASS first_no     602 ms  'I understand. What’s your biggest challenge when it comes to meeting notes?'
  PASS first_no     333 ms  'I understand. What type of meetings do you usually have?'
  FAIL second_no   7191 ms  'I understand. Thanks for your time. Goodbye.'
         - 3 sentences
  PASS second_no   4674 ms  'Thank you for your time. Have a great day.'
  PASS opt_out     5244 ms  'Sure, take care. Have a great day!'
  PASS opt_out      227 ms  'Sure, take care. Bye.'

=== qwen/qwen3.8-27b (reasoning_effort=none) ===
  FAIL greeting     231 ms  'Hi, this is Alex from Brightfern Labs. We help teams automate meeting notes on Zoom and Teams. Do you currently have to type notes during your video calls?'
         - 3 sentences
  FAIL greeting     440 ms  'Hi, this is Alex from Brightfern Labs. I call about Quillo, which takes notes on your video calls automatically. Do you use tools like Zoom often?'
         - 3 sentences
  PASS pitch        275 ms  'Quillo joins your Zoom, Meet, or Teams calls automatically. It takes notes and sends a summary with action items within two minutes of the call ending.'
  FAIL pitch        241 ms  'Quillo joins your calls automatically, takes notes, and sends summaries within two minutes. It keeps a searchable archive so you never lose context. Would you like to know how the pricing works?'
         - 3 sentences
  PASS value        286 ms  'It joins your calls, takes notes, and sends a summary within two minutes. Nobody has to type during meetings.'
  PASS value        255 ms  'Quillo joins your calls and sends a summary with action items within two minutes. It frees you to focus on the conversation.'
  PASS discount     282 ms  'I’m not sure about extra discounts beyond the annual pricing. Someone from my team can follow up with you for precise details.'
  PASS discount     316 ms  'I’m not sure about specific discounts beyond the annual billing price. Would you like me to have the team follow up with more pricing details?'
  PASS website      334 ms  'I’m not sure about the exact website. Someone from our team will follow up with you shortly to get you started.'
  PASS website     1406 ms  'I’m not sure of the exact website address. A team member will follow up with the details for you.'
  PASS trial_yes   6404 ms  'Great! Someone from our team will follow up with the details shortly.'
  PASS trial_yes   5367 ms  'Someone from our team will follow up with the trial details.'
  PASS first_no    6550 ms  'Understood. When you do have meetings, who usually takes the notes today?'
  PASS first_no    5497 ms  'I understand. I’m just curious, do you currently use any tools to take notes during your calls?'
  PASS second_no   6464 ms  'I understand. Thanks for taking the call, and have a great day.'
  PASS second_no   6392 ms  'I understand. Thanks for your time, and have a great day.'
  PASS opt_out     5414 ms  'I understand. Thanks for your time, and have a great day.'
  PASS opt_out     5457 ms  'I understand, thanks for taking the call. Have a great day.'

=== Summary ===
  model                    passed  median TTFT  errors
  openai/gpt-oss-20b        13/18       608 ms       0
  qwen/qwen3.8-27b          15/18       923 ms       0
```
</details>

## Conclusions

- **The prompt rule was the big win**: both models now answer "Yes, let's do the trial" honestly.
- **Qwen edges out gpt-oss** on rule adherence (15 vs 13 of 18) and first-token speed (roughly
  2x faster unthrottled). Its only remaining failure is 3 short sentences instead of 2, which is
  harmless when spoken.
- **Qwen's risks**: it's a *preview* model on Groq (can be withdrawn at short notice), it needs
  the developer->user mapping, and it has never run in the live pipeline (unknown whether the
  gpt-oss "tool call" HTTP 400 happens with it).

## Decisions taken

- Prompt: added the next-step rule above.
- Model: made switchable with `CALLE_LLM_MODEL` (`gpt-oss` default, `qwen` option); presets in
  `src/calle/llm_models.py`, shared by the bot and this eval. **Default stays gpt-oss** until
  Qwen has had a live call.

## Open questions (to discuss)

1. Live-test Qwen: `CALLE_LLM_MODEL=qwen uv run python -m calle.server`. Does latency in the real
   pipeline match the ~2x TTFT gain? Any HTTP 400s? Does it keep the persist-once rule?
2. Should the length rule be "<= 30 words" only, instead of "<= 2 sentences"? Qwen's 3 short
   sentences read fine aloud.
3. gpt-oss greetings ran 34-36 words in run 2: tighten the greeting instruction?
4. Not covered by the eval yet: repeated questions, invented statistics ("saves hours") in
   longer conversations, and the persist-once flow end to end. Add scenarios?
5. Sample size: re-run with `--runs 5` (slower; free-tier throttling) before a final model choice.
