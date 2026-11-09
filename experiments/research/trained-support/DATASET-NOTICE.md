# Trained support routing: dataset notice

The framework code is MIT licensed. The source datasets retain their own licenses and attribution; MIT does not replace those terms.

BANKING77 comes from **PolyAI-LDN/task-specific-datasets**, revision `57ec275d8078af65b7731c2a98be812d844a6d6b`, under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Please attribute Casanueva et al. (2020), *Efficient Intent Detection with Dual Sentence Encoders*, arXiv:2003.04807, and the PolyAI dataset release. The preserved upstream [CC BY 4.0 legal text](../../../docs/research/dataset-licenses/CC-BY-4.0.txt) is byte-identical to the acquired license.

The study uses the official 10,003-row training partition for disjoint training/development/target/protected/calibration roles and the official 3,080-row test partition after sealing. Published derived artifacts include independently trained encoder/head parameters, learned feature coordinates, role identifiers and labels, predictions, and evaluation records. The six-queue mapping and specialist ownership policy are this project's derived demonstration policy. They are not BANKING77 labels or observations of a bank's operations.

CLINC OOS comes from **clinc/oos-eval**, revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`, under **Creative Commons Attribution 3.0 Unported (CC BY 3.0)**. Please attribute Larson et al. (2019), *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP, ACL Anthology D19-1131, and the CLINC release. The preserved upstream [CC BY 3.0 legal text](../../../docs/research/dataset-licenses/CC-BY-3.0.txt) is byte-identical to the acquired license. Only its 1,000 `oos_test` queries are used, after sealing, as dispatch/confidence stress. Its labels establish no BANKING77 routing accuracy.

Raw acquired queries, raw dataset acquisition copies, acquisition HTML and incidental contact metadata are not redistributed here. Derived features and public model parameters are not a confidentiality or irreversible anonymization guarantee. Anyone obtaining the original observations should follow the upstream dataset terms.

[Sanitized source URLs, revisions, byte counts and SHA256 identities](DATASET-PROVENANCE.json) · [Frozen protocol](protocol.json) · [Heldout report](../../../docs/research/trained-applications.md).
