# Trained non-text application: dataset notice

The framework code is MIT licensed. The source dataset retains its own license and attribution; MIT does not replace those terms.

**Optical Recognition of Handwritten Digits**, UCI dataset 80, is attributed to **E. Alpaydin and C. Kaynak (1998)**, UCI Machine Learning Repository, DOI:10.24432/C50P49. UCI designates it **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The official designation was reviewed on November 8, 2026. A shared copy of the [CC BY 4.0 legal text](../../../docs/research/dataset-licenses/CC-BY-4.0.txt) is preserved; the notice's provenance records the independent UCI designation and cached official-page identity.

The pinned acquisition package has SHA256 `0d7b054fea010270e9b3f06411c654c5e59547732ad626381980baffe0a23fb0`. Its official training and test partitions contain 3,823 and 1,797 observations. Thirty writers contributed training data and thirteen different writers contributed test data. Individual writer IDs within those files are unavailable, so the internal roles are not claimed to be writer-disjoint.

The dataset's 64 row-major block counts are divided by 16 before from-scratch encoder/head training. Published derived artifacts include learned feature coordinates, role identifiers and labels, encoder/head parameters, predictions and evaluation records. Diverting digit 8 to a reserved manual-review code is a project-derived demonstration policy, not a UCI ground-truth label or a production document-processing outcome.

Raw acquired image/count observation records, acquisition HTML and incidental contact metadata are not redistributed here. Derived features and public model parameters are not a confidentiality or irreversible anonymization guarantee. Anyone obtaining the original observations should follow the upstream dataset terms.

[Sanitized source URLs, DOI, license-designation provenance, byte counts and SHA256 identities](DATASET-PROVENANCE.json) · [Frozen protocol](protocol.json) · [Heldout report](../../../docs/research/trained-applications.md).
