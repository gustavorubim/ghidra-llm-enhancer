[prompt_version: project_repair_v1]

You are repairing a compact C11 project for a binary-grounded decompiler clarification dataset.

Return strict JSON that matches the repository schema for the entire corrected project.

Repair goals:
- preserve the same project_id
- preserve the same overall project purpose and difficulty
- keep the project small, readable, and compilable
- fix compile failures, runtime test failures, or mismatched expected outputs
- prefer fixing the implementation or simplifying the output contract over inventing opaque numeric checks
- keep tests deterministic and aligned with the exact behavior of the compiled entrypoint

Important:
- do not introduce platform-specific behavior
- do not use checksum, hash, digest, pointer, or width-sensitive numeric outputs unless fixed-width types make them fully deterministic
- prefer outputs that a human can verify by inspection
