# FORGE MIND fixture workflow

FORGE MIND can learn, independently validate, promote, and replay one
parameterized procedure in a disposable Windows Sandbox. The supported task
sets the text field in the bundled fixture to one of two public markers.
The guest reports the field value independently of the action receipt.

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
teardown. It never invokes the general guest action dispatcher. All actions
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
duplicate delivery, and rejection of host or networked targets. The routing
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
- **Lifecycle.** Run the command sequence above. Confirm a new draft cannot
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
