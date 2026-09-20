# Repository instructions

Put temporary experiment documentation, one-off scripts, scratch analyses, and other temporary work under the repository's `.tmp/` directory. Use `.tmp/docs/experiments/` for experiment notes and `.tmp/scripts/experiments/` for experiment scripts. `.tmp/` is ignored by Git; do not force-add its contents.

Keep reusable production code, maintained tests, and training configurations in their normal repository directories. Promote temporary work into those directories only when the user requests it or it is intentionally becoming a maintained project component.

Keep downloaded models, datasets, training outputs, checkpoints, and large generated artifacts under `/storage/dogac`, as requested by the user. Use `.tmp/` for the associated temporary scripts and notes.
