# Laboratory motion review

Open reviewer.html in your browser. It works offline: choose a movement, play/pause,
step frames, scrub time and change the camera. Use 0.5x to inspect contact and pacing.
Download review JSON or tell us the movement and timestamp plus the desired change.

Direct video previews are in previews/dropper.mp4, pour.mp4 and stir.mp4.
These are actual 3D coordinate templates shown with a simple orthographic projection,
not photorealistic demonstrations. Scene units and durations are illustrative.
No fluid simulation, calibrated doses, hands, grip biomechanics or chemistry are supplied.
The drop and stream marks are cues, not predicted fluid dynamics.

Approve movement only after inspecting it. This does not approve scientific accuracy
or certify a model conditioning input. The viewer is not yet a depth/pose exporter.
Use build_previews.py to reproduce the geometry and videos. reviewer.template.html
contains the viewer source; reviewer.html embeds the exact motion data and its hash.

The expanded library has 24 named movement templates. source_motion_candidates.json
contains heuristic suggestions for source steps; those are not verified bindings.
Apparatus fitting, deformable membrane preparation, hands and source-specific
connections remain review/implementation work.
