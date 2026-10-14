PY := python3.12
VENV := .venv
BIN := $(VENV)/bin
EXE :=
ifeq ($(OS),Windows_NT)
PY := py -3.12
BIN := $(VENV)/Scripts
EXE := .exe
endif
PYTHON := $(BIN)/python$(EXE)
ifeq ($(OS),Windows_NT)
PYTHON := $(VENV)\Scripts\python.exe
endif
BOT ?= rookie
# `AS` is a GNU make BUILT-IN (the assembler, default `as`), so `AS ?= all`
# never fired and a plain `make spar BOT=rookie` ran `spar.py --as as`, which
# argparse rejects. `?=` only assigns when a variable is UNDEFINED, and make had
# already defined this one. Keep the documented `AS=defender` interface working
# by honouring AS only when it really came from the command line.
ROLE ?= all
ifeq ($(origin AS),command line)
ROLE := $(AS)
endif

.PHONY: install spar ui validate qualify submit test clean check-no-key

# --seed is REQUIRED: `uv venv` alone creates a venv with no pip.
install:
	uv venv --python 3.12 --seed $(VENV) || $(PY) -m venv $(VENV)
	$(PYTHON) -m pip install -q --upgrade pip
	$(PYTHON) -m pip install -q pytest
	@echo "ready. no api key needed, ever."

spar:
	$(PYTHON) spar.py --bot $(BOT) --as $(ROLE)

ui:
	$(PYTHON) -m kit.arena_ui.build_ui
	$(PYTHON) spar.py --bot $(BOT) --as $(ROLE) --ui
	$(PYTHON) -m kit.arena_ui.serve

# Always validate against the REAL exported world. Without --world the validator falls
# back to kit/world/fixture.py's ~40-page synthetic world, where every real anchor fails
# to resolve — 15 spurious failures that look like a broken deck and are not.
WORLD := $(firstword $(wildcard kit/world/*/manifest.json))

validate:
	@$(PYTHON) -c "import os,sys; p='$(WORLD)'; sys.exit(0 if os.path.isfile(p) else (print('no world exported - run make check-world') or 1))"
	$(PYTHON) validate_deck.py deck/deck.json deck/lineup.json --world $(dir $(WORLD))

validate-bots:
	@for b in rookie operator adversary; do \
		printf "%-12s " $$b; \
		$(PYTHON) validate_deck.py bots/$$b/deck.json bots/$$b/lineup.json \
			--world $(dir $(WORLD)) 2>&1 | tail -1; \
	done

# `qualify` used to run a `qualify.py` that was never written, writing a
# `submissions/radar.json` that NOTHING in either repo reads. It is not a
# missing dependency, it is a promise that was never wired up. The student's
# real conformance check is the public suite: `make test`.
qualify:
	@echo "make qualify: retired — nothing consumed submissions/radar.json."
	@echo "Your conformance check is 'make test' (the public suite)."
	@echo "Then: make validate && make submit TEAM=<your-team>"
	@exit 1

# `TEAM` is required by kit.submit. Check it in make itself so this target works
# with both POSIX shells and Windows cmd.exe (which has no `test -n` command).
submit: validate
ifeq ($(strip $(TEAM)),)
	@echo usage: make submit TEAM=your-team-name
	@exit 1
else
	$(PYTHON) -m kit.submit --team "$(TEAM)"
endif

test: check-no-key
	$(PYTHON) -m pytest tests/

# The referee in kit/ is a hash-synced copy of the arena's (CONTRACTS.md 2.4): students
# must be able to run the exact verifier that will judge them, or prosecution is guesswork.
check-referee:
	@$(PYTHON) -c "import os,sys; sys.exit('kit/referee missing - ask your instructor to run tools.sync_referee') if not os.path.isdir('kit/referee') else None; from kit.referee.rubric import CLASSES; from kit.referee.adjudicate import LOCAL_ONLY; print(f'referee: {len(CLASSES)} classes, local_only={LOCAL_ONLY}')"

# The world artifact is exported by the instructor; without it nothing can run.
check-world:
	@$(PYTHON) -c "import glob,json,sys; p=sorted(glob.glob('kit/world/*/manifest.json')); sys.exit('no world in kit/world/ - ask your instructor for the world artifact') if not p else None; m=json.load(open(p[-1])); c=m.get('counts',{}); print('world',m.get('world_id'),'-',c.get('__total__',sum(c.values())),'pages'); sys.exit('FAIL: truth.json must never ship to students') if glob.glob('kit/world/*/truth.json') else None"

doctor: check-no-key check-world check-referee validate
	@echo "ready to spar."

# A shipped gate, not a formality: the student kit must contain no model client and no
# API key. It is a real module with its own tests, not a grep — the grep version fired on
# the sandbox's own network-denial probe and on the injection fixtures that have to NAME
# the key to be realistic. Naming a secret is not leaking one; see kit/gate_no_key.py.
check-no-key:
	@$(PYTHON) -m kit.gate_no_key

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
