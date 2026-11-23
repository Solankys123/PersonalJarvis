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

## Windows verification and rollback

Before relying on the feature on Windows, inspect the generated sandbox XML,
confirm the fixture starts, then run the command sequence through chat and
local speech. Verify the first procedure stays a draft until the second guest
validates it and you explicitly promote it. Close the fixture during a run
and verify it fails safely. Confirm cancellation closes only the assigned
guest. Measure guest startup, peak RAM, paging, and replay latency on the
target hardware. The Linux test suite uses a deterministic fake screen and
does not launch Windows Sandbox.

Set `forge.enabled = false` and restart to disable the tool. To remove the
feature from a development install, switch to your prior checkout and refresh
its package metadata. The separate FORGE SQLite file can be retained as a
backup.
