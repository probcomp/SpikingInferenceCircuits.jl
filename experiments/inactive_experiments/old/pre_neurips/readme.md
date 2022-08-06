### `experiments/old/`
This directory contains code for experiments which relied on an earlier version
of the inference circuits.  (In particular, those in `src/circuits/inference/old`.)

I have not updated these experiments to either correctly pull in the old versions
of the inference circuit code, nor to use the new inference code.  If we end up wanting
to run these experiments again, we could either do it using an earlier git commit before the
changes to the inference circuits, or refactor a bit so we can either pull in the old inference code
in the current commit, or update the experiments to use the new inference circuits.