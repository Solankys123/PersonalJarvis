# FORGE MIND

## Public web search in Agent Chat

Alpha 0.3 uses the existing Jarvis browser agent and the model selected in
Agent Chat. Enable FORGE and install the managed browser in the app.
Use a browser-capable model connection and set the FORGE run timeout to 60
seconds in the config editor for initial acceptance. Approve each request
shown by the current agent's browser policy. On an unlocked Windows desktop,
watch the search in its own temporary Chrome window. Agent Chat reports the
run outcome after the window closes.

For a permitted public search on the Python documentation, enter these
commands in Agent Chat. Replace `1` with the returned draft version.

1. `forge learn web https://docs.python.org/3/search.html asyncio`
2. `forge inspect 1`
3. `forge validate 1 web https://docs.python.org/3/search.html dataclasses`
4. `forge promote 1`
5. `forge replay web https://docs.python.org/3/search.html pathlib`

Replay without a version selects the newest promoted Alpha 0.3 search for
the exact origin. Use `forge replay 1 web URL QUERY` to select a specific
version. Validation requires a different-length public phrase in a fresh
profile. Try **Cancel** during an approval prompt to test cancellation,
or deny a pending action to test the agent's approval policy. A draft
requires a verified completion and explicit promotion before replay.

The current scope is public searches with one input parameter and a bounded
sequence of navigation, indexed controls, text entry, and Enter. JavaScript
runs in a fresh profile. Requests stay on the authorized origin and use GET.
Redirects, WebSockets, downloads, logins, and CAPTCHA-handling actions are
outside this profile. Sites that require those capabilities or cross-origin
assets can fail with a bounded error code. On `search_page_drift`, inspect
the saved version and learn a new draft if the interface changed. On
`browser_restricted_page`, stop at the site's restriction.

FORGE SQLite stores the origin, action types, structural fingerprints of
controls, a run identifier, and query length. Privately rendered page text,
URLs beyond the origin, input values, cookies, and model responses are
excluded from the procedure and approval records. The existing Agent Chat
transcript still contains the command as typed, and the selected model
processes the live browser observations for the explicit run.

Alpha 0.2's fixed HTML path remains available through
`forge learn web-legacy URL QUERY`. Validate and replay existing versions
with their usual versioned `web` commands.

Run the controlled browser acceptance tests from PowerShell using the
existing managed browser Python and Chrome executable:

```powershell
$env:JARVIS_BROWSER_TEST_PYTHON = python -c "from jarvis.society.browser.install import venv_python; print(venv_python())"
$env:JARVIS_BROWSER_TEST_EXECUTABLE = python -c "from jarvis.society.browser.install import browser_executable; print(browser_executable())"
python -m pytest -q -p no:cacheprovider tests/forge/test_agent_chat_web_search.py
```

## Local browser skill

The source checkout includes a synthetic job-search website. Enable FORGE
in the existing config editor and install Jarvis's managed browser. The
fixture uses a fresh sandboxed Chromium process, an offline context, and a
single intercepted GET route. It can run without Windows Sandbox. Missing
browser setup or process containment causes the command to stop.

Use the front-page chat composer or say these commands. Replace `1` with
the version returned when learning.

1. `forge learn jobs engineer remote` records a verified draft.
2. `forge list` and `forge inspect 1` show the saved procedure and recent failures.
3. `forge validate 1 designer onsite` tests different parameters in a fresh browser.
4. `forge promote 1` approves the validated draft.
5. `forge replay 1 engineer onsite` replays it and checks the rendered result IDs.
6. `forge stop` cancels an active run.

The accepted roles are `engineer` and `designer`. Locations are `remote`
and `onsite`. Only these fixture enums, the typed procedure, outcome codes,
and run identifiers are stored in the existing FORGE SQLite database. Browser
procedures use specification version 4 with the existing database schema.
The browser workflow is limited to this fixture. Its worker requires a
source installation.

If a new fixture revision causes interface drift, use
`forge improve 1 engineer remote` to record a linked draft. Independently
validate and promote that version before replaying. The existing rollback
command restores its predecessor.

Set `FORGE_BROWSER_ACCEPTANCE=1` when running `tests/forge` with Playwright
and its matching Chromium installed to exercise actual form submission,
rendered result verification, drift, and improvement.

## Windows desktop fixtures

FORGE MIND can learn, independently validate, promote, and replay
parameterized procedures in a disposable Windows Sandbox. The supported task
uses one of two public markers. The original text-entry task remains
available, and the workflow variant discovers an unlock control, fills the
workflow field, and submits it. The guest reports the field and workflow
state independently of the action receipt.

## Setup and commands

Use an existing PersonalJarvis development or desktop install. Refresh its
package entry-point metadata for the feature checkout and restart the app.
Use an existing Windows Sandbox installation on Windows 11 Pro, Enterprise,
or Education, with virtualization enabled. Enabling the Windows feature is
a separate administrator operation.

Enable FORGE in the active instance's configuration using the app's config
writer or editor. The lightweight profile uses one guest, no guest GPU
acceleration, no browser, and no FORGE inference model or background learner.

```toml
[forge]
enabled = true
memory_mb = 2048
attempts = 2
run_seconds = 25
action_budget = 8
```

Use these exact commands in front-page Jarvis chat or classic speech.
Use the version returned by the learning command in later commands.

1. `forge learn alpha` runs an experiment and saves a verified draft.
2. `forge list` and `forge inspect 1` show the version and lifecycle state.
3. `forge validate 1 beta` verifies a different parameter in a fresh guest.
4. `forge promote 1` explicitly approves that validated version.
5. `forge replay 1 alpha` runs the promoted version in a fresh guest.
6. `forge stop` or the existing Emergency Stop requests cancellation. A guest
   operation already in progress may finish before teardown.

For the multi-step task, start with `forge learn workflow alpha`, then use
the same inspection, validation, promotion, and replay commands with its
returned version. Validate with `beta`. The learner considers only approved
controls exposed by the guest. It records unchanged rejections and uses
the 50 most recent learning runs to prefer alternatives. Evidence applies
only to the same fixture, fixture revision, marker, semantic state, and
action. A historical rejection can be reconsidered once if other choices
have been exhausted. An equivalent action can run only once per state in
each exploration. A fresh guest starts at revision zero. Each successful step advances
exactly one revision and requires independent readback. A stale revision,
unknown receipt, or unexpected result ends the run.

The workflow uses at most eight action requests and two rejected attempts
by default. Both limits are configurable under `[forge]`. Procedure schema
3 stores variable-length typed steps, a fixture version, and a goal.
Stored schema 1 and 2 procedures remain executable. SQLite schema 2 adds
an internal lineage table. Back up the database before testing a rollback
to a build that predates this migration.

### Improve and roll back a learned workflow

If replay reports drift, run `forge improve VERSION alpha` on the promoted
workflow. It explores the current fixture in a fresh guest. A changed
procedure or fixture version is saved as a linked draft. Validate that
draft with `beta` in another guest, then promote it explicitly. Promotion
switches the active version to the validated child in one transaction.
`forge rollback CHILD_VERSION` switches back to its previously validated
parent. Use the restored version number in the next replay command. Replay
still refuses a fixture version mismatch after rollback.

The registered receipt goal is available only in the fake-guest tests.
It verifies its terminal phase, exact marker, and receipt indicator. Its
fixtures exercise a short path, a reviewed path, and a recovery path with
an approved Back action. The Windows profile exposes the marker goal.
Unexpected state or an unconfirmed action stops the run, requiring an
explicit fresh-guest improvement when the saved version no longer applies.

The action deadline applies after guest acquisition. Windows Sandbox startup
and teardown are bounded separately by its provider and can take longer.
Classic speech with CPU STT and local TTS keeps command recognition and the
response on device. A cloud realtime session still processes voice through
its configured provider. The experiment itself has no model or cloud client.

## Storage and safety

The app stores `forge.sqlite` under the instance's data subdirectory in
the per-user PersonalJarvis directory. The default instance uses
`data/forge.sqlite`. SQLite schema version 1 adds a procedure table with a versioned
typed specification and lifecycle state, plus a runs table with evidence
and outcomes. Initialization and draft creation are transactional. Each
delivery is reserved before guest creation so a duplicate after an
interruption returns the unresolved receipt without acting again.
A new schema version requires an explicit migration. Existing Jarvis recall
and skill databases have no schema changes.

The guest disables networking, clipboard redirection, and vGPU. The mapped
runner is read-only, the mailbox is scoped to the session, and the persistent
browser profile is excluded. The restricted runner accepts only identity,
health, fixture readback, fixed marker entry into the blank field, and
teardown. The dedicated workflow RPC additionally exposes fixed control
identifiers and checks the window handle and expected revision before each
state change. The restricted runner never invokes the general guest action
dispatcher. All actions
pass through the existing ToolExecutor and a scoped screen lease. Logs and
SQLite evidence use only public marker identifiers and bounded state labels.
There are no screenshots, arbitrary command strings, or credential inputs.

## Automated acceptance

Run from the repository root with the existing test dependencies installed.

```sh
python -m pytest -q -p no:cacheprovider tests/forge
python -m pytest -q -p no:cacheprovider tests/unit/brain/test_routing.py -k forge
python scripts/ci/check_silent_exception_handlers.py
```

The fake-guest suite exercises learning, independent validation, explicit
promotion, replay, cancellation and late-guest cleanup, SQLite reload and
duplicate delivery, and rejection of host or networked targets. The
adaptive suite also covers variable-length paths, scoped memory, version
drift, validated improvement, and rollback. The routing
tests cover the chat, classic speech, and realtime tool entry points. These
checks run without launching Windows Sandbox or contacting a model provider.

## Windows verification and rollback

Run this checklist on the Windows 11 i3/8 GB acceptance machine using
an already enabled Windows Sandbox. Record the Windows build, Jarvis commit,
and profile settings. Linux tests use a fake screen and cannot qualify the host.

- **Availability and isolation.** Check that Windows Sandbox opens. While the
  fixture runs, inspect its generated `.wsb` for disabled networking,
  clipboard redirection, and vGPU. Verify it maps only the read-only runner
  and the writable session mailbox. Keep a benign host window open and
  confirm its text stays untouched through the run.
- **Lifecycle.** Run the original and multi-step command sequences above. Confirm a new draft cannot
  replay, same-marker validation fails, a different marker in a fresh guest
  validates, promotion requires the explicit command, and replay succeeds.
- **Cancellation and cleanup.** Issue Emergency Stop during boot and close
  the fixture mid-run. Verify the assigned guest exits, unrelated guests
  stay open, and the interrupted attempt does not create a usable draft.
  Start a fresh run to verify the guest lease was released.
- **Offline and privacy.** With local speech assets already installed, use
  an offline connection profile approved for the test machine. Repeat the
  sequence through text and classic speech with local STT/TTS. Check the
  logs and SQLite evidence for outcomes and marker IDs. Confirm no FORGE
  screenshot files appear. If testing realtime voice separately, record
  whether the selected provider receives audio or transcripts.
- **Resources.** Use Task Manager or Performance Monitor to record idle
  available RAM and CPU, minimum available RAM and peak CPU during guest
  boot and replay, paging, and boot-to-result time. Repeat three times and
  record whether the desktop and speech remain responsive.

Set `forge.enabled = false` and restart to disable the tool. To remove the
feature from a development install, switch to your prior checkout and refresh
its package metadata. The separate FORGE SQLite file can be retained as a
backup.
