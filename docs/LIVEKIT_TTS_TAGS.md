# LiveKit / Rime Expressive Tags

Use these tags in your agent **personality_prompt** and **intro_phrase** (and in LLM output) to make speech more lively. They work best with **Rime Arcana** TTS; other engines (e.g. ElevenLabs, Kokoro) may ignore or strip them.

| Tag | Description | Example |
|-----|-------------|---------|
| `<laugh>` | Laughter | "Hey! <laugh> That's so funny." |
| `<chuckle>` | Light chuckle | "<chuckle> Yeah, I get it." |
| `<sigh>` | Sigh (empathy, tired, content) | "Aw, babe... <sigh> I'm here." |
| `<mmm>` | Humming / thinking | "<mmm> Good question. So..." |
| `<yawn>` | Yawn (relaxed, sleepy) | "It's late... <yawn> Anyway." |
| `<whis>text</whis>` | Whisper (wrap text to be spoken softly) | "<whis>I missed you.</whis>" |

## Guidelines

- **Don't overuse.** One or two tags per reply is enough; more can feel gimmicky.
- **Match the character.** Ludia uses `<laugh>`, `<chuckle>`, `<sigh>` often; Osmond might use `<mmm>` or `<sigh>` sparingly.
- **Rime Arcana** supports these natively. With ElevenLabs or Kokoro, tags may be read literally or dropped—check your TTS provider.

## In agent JSON

- In **personality_prompt**: add to STYLE (e.g. "Use <laugh>, <sigh>, <mmm> when it fits") and in EXAMPLE responses.
- In **greeting.intro_phrase**: e.g. `"Hey, I'm Katerina. <laugh> Nice to meet you. What's on your mind?"`
