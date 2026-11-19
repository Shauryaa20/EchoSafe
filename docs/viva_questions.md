# Viva questions

**Why not call this diagnosis?** The model sees text patterns from non-clinical datasets and cannot observe a person's full context or state.

**Why separate the tasks?** GoEmotions and Dreaddit do not share labels or records. A shared architecture can be reused without pretending the datasets are jointly annotated.

**Why context attention?** Earlier messages can change interpretation, but masking prevents padded turns from affecting the result.

**What does uncertainty mean?** It describes model confidence and calibration, not psychological certainty.
