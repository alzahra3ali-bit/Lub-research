Comparing Pretrained Arabic Language Models on Intended-Meaning Classification of Saudi Dialect Text

Course project (Deep Learning). We build a four-class Saudi-dialect dataset (Positive, Negative, Neutral, Ironic) from three sources, label 1,200 texts by hand, and compare four pretrained Arabic encoders (AraBERT, MARBERT, MARBERTv2, SaudiBERT) with stratified 5-fold cross-validation and leave-one-source-out (LOSO) testing.


Data

Source	Original labels	Reference
Sa'7r	ironic / not ironic	AlMazrua et al., OSACT 2022
Telecom (AraCust)	positive / negative	Almuqren and Cristea, PeerJ CS 2021
Bank (Saudi Bank Sentiment Dataset, 12,048 tweets about four Saudi banks)	positive / negative / neutral	Alqahtani et al., ICNLSP 2022; https://github.com/iwan-rg/Saudi-Bank-Sentiment-Dataset (check its license)

Raw datasets are not redistributed unless their licenses allow it. See data/README.md for where to download them.

Gold set (manual labels, 1,200 texts): annotator 1 labeled all rows, annotator 2 labeled 200 (Cohen's kappa = 0.70).

Source	Positive	Negative	Neutral	Ironic
Bank	61	225	140	46
Sa'7r	60	68	42	238
Telecom	92	145	39	44
Total	213	438	221	328

Label definitions: Positive = compliment or satisfaction; Negative = complaint; Ironic = surface wording opposite to the intended meaning; Neutral = no evaluative or hidden meaning.


Main results (stratified 5-fold CV on the 1,200 gold texts)

Model	Accuracy	Macro-F1	95% bootstrap CI (OOF Macro-F1)
Majority class	0.365 ± 0.002	0.134 ± 0.001	[0.126, 0.141]
TF-IDF + Logistic Regression	0.692 ± 0.016	0.666 ± 0.016	[0.642, 0.695]
AraBERT	0.784 ± 0.025	0.771 ± 0.027	[0.747, 0.797]
MARBERT	0.791 ± 0.024	0.775 ± 0.023	[0.749, 0.800]
MARBERTv2	0.771 ± 0.033	0.755 ± 0.035	[0.731, 0.780]
SaudiBERT	0.802 ± 0.030	0.787 ± 0.031	[0.761, 0.811]

The confidence intervals of the four transformers overlap, so differences between them are small. LOSO results, per-class and per-source scores, and the pseudo-labeling ablation (no improvement) are in results/.

Reproducibility details

	•	Random seed 42 (fold seed = 42 + fold index); fold assignments are saved in results/fold_assignments.csv.
	•	Hyperparameters: max length 128, AdamW, learning rate 2e-5, batch size 16, up to 4 epochs, early stopping (patience 2) on an inner dev split (10% of each training part), class-weighted cross-entropy.
	•	Models: aubmindlab/bert-base-arabertv02-twitter, UBC-NLP/MARBERT, UBC-NLP/MARBERTv2, faisalq/SaudiBERT.
	•	Tested with Python 3.13, torch 2.11, transformers 4.57.6, scikit-learn 1.6.1 (Google Colab).

