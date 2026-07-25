# Cover Letter

Dear Editor-in-Chief of Applied Intelligence,

We are pleased to submit our manuscript entitled "Diagnosing and Mitigating Epistemic Uncertainty Degeneracy in Binary Evidence Deep Learning: A Case Study on Rainfall Occurrence Prediction" for consideration for publication in Applied Intelligence.

Evidence Deep Learning (EDL) has become a widely adopted method for uncertainty-aware classification, placing a Dirichlet distribution over class probabilities to produce aleatoric/epistemic uncertainty decomposition in a single forward pass. However, recent theoretical work (Shen et al., NeurIPS 2024; Jürgens et al., ICML 2024) has questioned whether EDL's epistemic uncertainty faithfully captures model ignorance. These critiques, while important, remain qualitative or identifiability-theoretic and do not provide practitioners with a computable diagnostic criterion.

Our manuscript addresses this gap with the following contributions:

1. **A quantitative diagnostic criterion** (Theorem 3): we prove that for binary EDL, epistemic uncertainty satisfies $I \approx 1/(2S)$, making it a monotone function of total evidence $S$ alone. The rank correlation between $I$ and $1/S$ exceeds 0.999 on real data, yielding a simple, computable check: if $H_E \propto 1/S$ on a validation set, epistemic uncertainty is degenerate.

2. **A comprehensive empirical diagnostic protocol** including meteorological skill scores (POD, FAR, CSI, HSS), cost-loss decision analysis, selective prediction, and four types of OOD evaluation on 142,193 station-days from 49 Australian stations. We honestly report that EDL is not the best predictor (LSTM and GRU achieve slightly higher accuracy), and analyze why.

3. **A practical mitigation strategy** via Mondrian conformal prediction, which provides distribution-free group-conditional coverage guarantees (0.9499 at 27.9% abstention) as a lightweight post-hoc wrapper.

4. **Three actionable guidelines for EDL practitioners**, derived from our theoretical and empirical findings.

We believe this work is well-suited for Applied Intelligence because it addresses the journal's focus on "intelligent systems methodology and its application in solving real-world complex problems"—specifically, providing an intelligent diagnostic framework for uncertainty quantification in operational meteorology. The binary classification setting is the simplest case where the degeneracy is analytically tractable, and the diagnostic criterion is directly transferable to other binary EDL applications.

We emphasize that this paper does not claim to propose a new state-of-the-art rainfall predictor. Rather, it provides a diagnostic framework for evaluating whether EDL's uncertainty claims hold in a given application, and a practical mitigation strategy when they do not. All 350+ experimental numbers reported in the paper are traceable to released result files; the full source code, pre-trained checkpoints, and reproduction guide are available at:

https://github.com/mingyi0818/Evidence_Rainfall

This work was supported by the Guangdong Provincial Undergraduate Higher Education Teaching Reform Project (Grant No. 粤教高函〔2024〕9-989). The authors declare no competing financial or non-financial interests. This manuscript has not been published elsewhere and is not under consideration by any other journal.

We look forward to your favorable consideration.

Sincerely,

Yafen Feng (Corresponding Author)
School of Geography and Tourism, Jiaying University
Key Laboratory of Surface Environment and Green Development in Northeast Guangdong
Email: fyf81@163.com

On behalf of all co-authors:
Jingyuan Zeng, Ming Zeng, Jianghong Guo, Chuanxian Jiang, Yafen Feng
