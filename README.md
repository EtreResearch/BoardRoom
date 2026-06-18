# BoardRoom

Four AI executives — CEO, CTO, Head Architect, and Head of Product — debate your
business idea in your terminal and hand you a verdict. The conversation streams
live, so you can watch them think.

<img width="1476" alt="BoardRoom discussion" src="https://github.com/user-attachments/assets/fcbfadcf-1199-4780-b7aa-00a2fea1ac8f" />

<img width="1821" alt="BoardRoom verdict" src="https://github.com/user-attachments/assets/75074fc2-2df5-43eb-a6bc-a83d3f06c5c8" />

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your ANTHROPIC_API_KEY
```

## Run

```bash
python boardroom.py --tui "An app that plans toddler meals with AI."
```

Watch the executives debate, then read their verdict. That's it.

## Common options

| Flag | What it does |
|------|--------------|
| `--tui` | Full-screen interface (recommended) |
| `--rounds N` | Debate rounds before the verdict (default: 3) |
| `--model ID` | Use a different model — e.g. a cheaper one |
| `--verdict MODE` | Change how the verdict is presented |

In the TUI: `q` quit · `s` save the transcript · `i` add a note mid-debate.

See **[docs/details.md](docs/details.md)** for every flag, the verdict modes, and how it all works.

## Customize the panel

Edit `agents.yaml` to change who's in the room. Each executive has a name, a
color, and a short description of their point of view. Add or remove as many
as you like.

## Roadmap

- [ ] Improved UI
- [ ] Support for open-source / local model providers
- [ ] Built-in token compression
- [ ] Demo video
