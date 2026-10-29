Historical implementation retained for inspection only. These files use a
Transformer encoder, augmented dynamics and evidential classification, and do
not implement the two base models in paper_latentODE.pdf. Existing historical
models/ and results/ artifacts are unrelated to results/paper/.

The active implementation is src/; invoke python -m src.train or the scripts/
entry points from the repository root. No checkpoint compatibility is claimed.
