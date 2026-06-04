# TRIBE v2: A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience

> **Paper:** arXiv 2605.04326v1
> **Authors:** Stéphane d'Ascoli, Jérémy Rapin, Yohann Benchetrit, Teon Brooks, et al. (FAIR at Meta)
> **Date:** May 7, 2026
> **Code:** https://github.com/facebookresearch/tribev2
> **Weights:** https://huggingface.co/facebook/tribev2
> **Demo:** https://aidemos.atmeta.com/tribev2

---



---
<!-- Page 1 -->

A foundation model of vision, audition, and
language for in-silico neuroscience
Stéphane d’Ascoli1, Jérémy Rapin1, Yohann Benchetrit1, Teon Brooks1, Katelyn Begany1, Joséphine
Raugel1,2, Hubert Banville1, Jean-Rémi King1
1FAIR at Meta, 2Laboratoire de Neurosciences Cognitives et Computationnelles, Ecole Normale
Supérieure - PSL
Cognitive neuroscience is fragmented into specialized models, each tailored to specific experimental
paradigms, hence preventing a unified model of cognition in the human brain. Here, we introduce
TRIBE v2, a tri-modal (video, audio and language) foundation model capable of predicting human
brain activity in a variety of naturalistic and experimental conditions. Leveraging a unified dataset
of over 1,000 hours of fMRI across 720 subjects, we demonstrate that our model accurately predicts
high-resolution brain responses for novel stimuli, tasks and subjects, superseding traditional linear
encoding models, delivering several-fold improvements in accuracy. Critically, TRIBE v2 enables in
silico experimentation: tested on seminal visual and neuro-linguistic paradigms, it recovers a variety
of results established by decades of empirical research. Finally, by extracting interpretable latent
features, TRIBE v2 reveals the fine-grained topography of multisensory integration. These results
establish artificial intelligence as a unifying framework for exploring the functional organization of the
human brain.
Date: May 7, 2026
Correspondence: sdascoli@meta.com and jeanremi@meta.com
Code: https://github.com/facebookresearch/tribev2
Weights: https://huggingface.co/facebook/tribev2
Demo: https://aidemos.atmeta.com/tribev2
1
## Introduction
Neuroscience has historically progressed through a highly specialized study of cognitive functions and their
underlying neural substrates. In the domain of vision, for instance, research has systematically mapped the
representations of motion to area V5 (Shadlen and Newsome, 2001), faces to the fusiform gyrus (Kanwisher
and Yovel, 2006), and written language to the visual word-form area (Dehaene and Cohen, 2011). While this
"divide-and-conquer" approach has yielded deep insights, the resulting landscape remains fragmented and
difficult to synthesize. Understanding how neuronal assemblies represent and integrate information to form a
coherent model of the surrounding world remains a fundamental challenge (Mathis et al., 2024; Yamins and
DiCarlo, 2016; Richards et al., 2019).
arXiv:2605.04326v1  [q-bio.NC]  5 May 2026
The fast progress of foundational models for language, vision, and audio offers a promising resolution to this
challenge. Indeed, increasing evidence suggests that the latent hierarchies of deep neural networks exhibit a
striking convergence with the representational geometry of the primate brain (Kay et al., 2008; Yamins et al.,
2014; Van Gerven and Bohte, 2017; Kriegeskorte, 2015). This alignment effectively enables direct prediction
of brain responses to natural images (Yang et al., 2023; Adeli et al., 2023; Nguyen et al., 2023; Ozcelik and
VanRullen, 2023; Scotti et al., 2024), videos (Lahner et al., 2025; Sartzetaki et al., 2024), sound (Kell et al.,
2018; Millet et al., 2022; Giordano et al., 2023) and text (Mitchell et al., 2008; Huth et al., 2016; Toneva and
Wehbe, 2019; Schrimpf et al., 2018; Caucheteux and King, 2022) by a linear transformation of the neural
network’s latent representations.
The representational alignment between brains and algorithms delineates a path toward a foundation model
of human brain function – derived not from first principles, but from the direct mapping of large amounts of
brain responses to pretrained AI architectures. For this goal to be successful, four essential criteria must be
1


---
<!-- Page 2 -->

![Figure 1 (p2, 1929×1165px)](tribe_v2_figures/figure_01_p2.png)

Figure 1 Approach. A. The brain activity of healthy volunteers was recorded with functional Magnetic Resonance
Imaging (fMRI), while they were presented to naturalistic (movies, podcasts) and experimental conditions (e.g. flashed
objects, isolated words). B. TRIBE v2 is trained to predict high-resolution fMRI from any audio, video, and/or text
stimuli, using embeddings from pretrained AI models Grattafiori et al. (2024); Chung et al. (2021); Assran et al. (2025).
C. Example of the average brain response (mean across 176 subjects) to a movie clip (Van Essen et al., 2013), and the
corresponding zero-shot prediction from TRIBE v2.
met: first, integration, where the model captures whole-brain responses across a vast repertoire of experimental
conditions; second, performance, reaching or exceeding the predictive accuracy of traditional analytical
pipelines tailored to specific neural response patterns; third, generalization, allowing the model to “zero-shot”
generalize to novel experimental conditions and thus equip researchers with a tool to improve the design,
data-efficiency, and analysis of their experiments; and finally, interpretability, providing a mechanistic toolkit
to decompose the organization of cognitive functions and neural representations.
Here, we present TRIBE v2, a tri-modal (audio, video, and language) foundation model for human brain
activity. Building on the v1 architecture (d’Ascoli et al., 2026) – which achieved state-of-the-art performance
in the 2025 Algonauts challenge (Gifford et al., 2024; St-Laurent et al., 2023) – we here scale this approach to
high-resolution fMRI and evaluate it on a variety of "deep-" and "wide-" datasets, together encompassing over
1,000 hours of fMRI recordings across 720 subjects table 1. We demonstrate that TRIBE v2: (1) accurately
predicts cortical responses across diverse naturalistic and experimental conditions; (2) enables rapid, in silico
hypothesis testing; and (3) provides a unified framework to accelerate neuroscientific discovery.
2
## Results
2.1
Encoding performance across naturalistic tasks
To evaluate TRIBE v2, we first assess its ability to reliably predict the brain responses to naturalistic stimuli
on subjects used during training. Specifically, we hold out fMRI responses to naturalistic conditions and
correlate them with TRIBE v2’s predictions given video, sound and texts (section 5.3). The results show that
a wide variety of cortical and subcortical regions are predicted above chance (figure 2A-C; see section 5.6
for details on statistical tests). As expected, the spatial distribution of these predictions vary with the task:
2


---
<!-- Page 3 -->

Dataset
Mode
Modalities
Device
Subjects
Sessions
fMRI
(h)
Video
(h)
Audio
(h)
Sentences
CNeuroMod (St-Laurent et al., 2023)
Train
A+V+T
3T
4
1408
268.7
64.5
64.5
54k
BoldMoments (Lahner et al., 2024)
Train
A+V
3T
10
520
61.9
33.2
32.2
-
Lebel2023 (LeBel et al., 2023)
Train
A+T
3T
8
432
85.8
-
16.1
5k
Wen2017 (Wen et al., 2018)
Train
V
3T
3
258
35.2
3.1
-
-
Subtotal
Train
25
2618
451.6
100.7
112.7
59k
NNDb (Aliko et al., 2020)
Test
A+V+T
3T
86
86
160.6
19.4
19.4
3k
LPP (Li et al., 2022)
Test
A+T
3T
112
1008
180.2
-
4.8
4k
Narratives (Nastase et al., 2021)
Test
A+T
3T
321
678
146.6
-
4.4
4k
HCP (Van Essen et al., 2013)
Test
A+V+T
7T
176
704
178.7
1.0
1.0
496
Subtotal
Test
695
2476
666.1
20.5
29.7
11k
Total
All
720
5094
1117.7
121.1
142.4
71k
Table 1 Characteristics of the datasets. Wide datasets prioritize population-level scale and generalizability, whereas deep
datasets prioritize individual-level granularity and precision.
podcast listening leads to peak correlation in the temporal cortices (Lebel2023); video watching (Wen2017 and
BoldMoments) leads to peak correlations in the visual cortices; and multimodal stimuli (Courtois NeuroMod)
combines these two peaks, yielding statistically significant predictions across most of the cortex. These
relatively high cortical scores contrast with those achieved in subcortical regions, which tend to be more
uniform and lower by two to three folds, but remain significant in most areas. Overall, these predictive
performance scores suggest that TRIBE v2 adequately captures a vast repertoire of naturalistic conditions.
2.2
Comparison to baselines
To contextualize the performance of TRIBE v2, we then compare it against a high-capacity implementation
of the Finite Impulse Response (FIR) model (Dale, 1999) – the current gold standard for voxel-wise encoding.
Traditional FIR models are typically constrained to single-subject analysis, hand-crafted features and small-
scale datasets. To provide a more rigorous baseline, we here developed a "Deep FIR" encoder optimized
via stochastic gradient descent and fed with the same state-of-the-art pretrained embeddings used by
TRIBE v2 (section 5.3). This ensures that the performance gap reflects the architectural advantage of our
transformer-based integration rather than differences in input features or optimization scale. Across all
datasets, TRIBE v2 significantly outperforming the optimized linear baseline (q(FDR) < 10−4, t-test across
subjects), demonstrating the advantage of deep nonlinear methods to model brain responses (figure 2D). We
further characterize the scalability of our architecture by analyzing performance as a function of data volume.
We observe a log-linear increase in encoding accuracy across the Courtois NeuroMod dataset without any
plateau (figure 2E), suggesting that TRIBE v2 is uniquely positioned to benefit from the ongoing expansion of
neuroimaging repositories. We first showed this scaling capability in the Algonauts 2025 competition, where
an earlier iteration of our TRIBE v2 architecture achieved first place out of 263 teams, establishing a new
state-of-the-art for the field (table 2).
2.3
Generalization to new subjects
To assess whether TRIBE v2 can generalize to new subjects, we curated four naturalistic studies for testing
consisting of many subjects recorded for a short amount of time (table 1). We first evaluate the ability
of TRIBE v2 to predict responses without any retraining, by computing the correlation Rgroup between
its predictions and the group-averaged response (see section 5.3 for implementation details). We compare
this group-level correlation with the “group-predictivity” of subject i, i.e. the correlation between subject
i’s response and the group response. Strikingly, across all datasets considered, TRIBE v2’s predictions
provide a more accurate estimation of the group response than the recording of most individual subjects
(figure 3A-B). The gap is most notable in the Human Connectome Project (HCP) dataset, which features the
best signal-to-noise ratio (it is the only dataset recorded with a 7T scanner, and features the most repetitions
of each stimulus): here, TRIBE v2 boasts an Rgroup near 0.4, a two-fold improvement over the median
subject’s group-predictivity.
3


---
<!-- Page 4 -->

![Figure 2 (p4, 8713×5487px)](tribe_v2_figures/figure_02_p4.png)

Figure 2 TRIBE v2 predicts fMRI responses accurately across the whole brain. A. Encoding scores in various important
regions of the cortex as defined by the HCP parcellation (see App. figure 8 for more details), color-coded in the cortical
surface on the left. Error bars denote Standard Error of the Mean (SEM) across the participants of all studies. B.
Encoding scores in the 8 subcortical regions considered, color-coded in the figure on the left. Error bars denote SEM
across the participants of all studies. C. Subject-averaged encoding scores across cortical and subcortical regions for
the four conditions in the training dataset. D. Average encoding scores of individual subject (dots) for TRIBE v2 and
a classic linear baseline. E. Scaling laws of brain encoding. Average encoding score of TRIBE v2 on the Algonauts
dataset, as we increase the number of hours of training data used per participant. Error bar indicates the SEM across
subjects.
2.4
Finetuning to improve individual brain modeling
While these results confirm that TRIBE v2 can model subject-averaged responses of unseen studies, it remains
to be tested whether it can model subject specificities with the limited data per participant available in these
broad studies. For this, we hold out half of the data for each participant (at most one hour) and finetune
TRIBE v2 for one epoch on each subject (see section 5.4). This significantly increases the encoding scores in
all datasets considered (q(FDR) < 10−4, t-test across subjects), and leads to a two- to four-fold improvement
over a linear encoder trained from scratch on each subject (figure 3C-D).
Overall, these results demonstrate that TRIBE v2 can be used to predict brain responses for unseen participants,
either by modelling the group-averaged response in a zero-shot manner or by using a small amount of fine-tuning
data to model subject specificities. Crucially, the fact that TRIBE v2 predicts group-averaged responses better
than individual responses suggests that it could readily be used for piloting naturalistic studies. However,
how would such performance hold for non-naturalistic studies?
2.5
In-silico experiments: vision
While the studies considered here all involve naturalistic perception tasks, the vast majority of neuroscientific
experiments involves controlled stimuli, aimed at isolating a specific phenomenon in the brain. An example
for this is the Individual Brain Charting (IBC) dataset (Pinho et al., 2018), which consists in a battery
4


---
<!-- Page 5 -->

![Figure 3 (p5, 3817×2402px)](tribe_v2_figures/figure_03_p5.png)

Figure 3 TRIBE v2 generalizes zero-shot to new tasks and subjects, and can be finetuned on a small amount of data to improve
individualized predictions. A. Left: Pearson correlation between (i) the brain response to speech of an individual subject
k and (ii) the average brain response of the rest of the cohort. Each dot represents a subject. The bars represent
the correlation between the average cohort response and the zero-shot predictions from TRIBE v2 (green) and its
linear counterpart (red). Right: cortical map of the scores achieved, averaged across datasets. B. Same as A but for
videos instead of speech. C. Left: encoding scores of TRIBE v2 for all subjects of the unseen speech datasets (each
dot represents a subject and the horizontal lines represent the mean). We compare the results obtained with a linear
model (either zero-shot or trained from scratch) to those obtained with TRIBE v2 (either zero-shot or finetuned on
half of the unseen subjects’ data). Right: cortical map of the scores achieved, averaged across datasets. D. Same as C
but for videos instead of speech.
of classic experiments designed to obtain functional localizers, e.g. to identify the brain areas involved in
face recognition or language processing. To test whether TRIBE v2 is reliable in such controlled settings,
we replicate some of these experiments in silico to compare its predictions with actual results achieved by
averaging the responses from participants of the IBC dataset.
We begin by replicating classic visual functional localizers from IBC: various categories of images (figure 4B) are
flashed for 1 second every 8 seconds. Note that although this type of protocol is far from naturalistic, TRIBE
v2 yields the expected dynamical response (figure 4A): activity increases in the ventral visual stream, and peaks
5 seconds after the stimulus onset, as expected from the hemodynamical delay. Various contrast maps are then
computed following the IBC protocol, as detailed in section 5.9. A qualitative match is observed between the
areas predicted by TRIBE v2 (figure 4C) and those measured in the original experiments (figure 4D), which
is quantified by a significant spatial correlation between the corresponding maps (figure 4E). In particular,
well-known responsive areas are recovered: the fusiform face area (FFA) for faces, the parahippocampal place
area (PPA) for places, the extrastriate body area (EBA) for bodies and the visual word-form area (VWFA)
for written characters (figure 4E).
5


---
<!-- Page 6 -->

![Figure 4 (p6, 5979×635px)](tribe_v2_figures/figure_04_p6.png)
![Figure 5 (p6, 7057×5211px)](tribe_v2_figures/figure_05_p6.png)

Figure 4 TRIBE v2 recovers, in-silico, a variety of findings in visual neuroscience. TRIBE v2 is tested on various visual
functional localizers, each extracted from the Individual Brain Charting (IBC) dataset (Pinho et al., 2018). Experimental
images are flashed for 1 second, with an interval of 8 seconds, and a general linear model (GLM) is fit on the predicted
time-series to obtain z-scored contrast maps (see section 5.9 for details). A. Mean evoked response to the images of
the FaceBody task. B. Examples of images from each category. The images are taken from the FaceBody and Visu
tasks from the IBC dataset. C. Contrast maps obtained from TRIBE’s predictions (see section 5.9). D. Contrast maps
computed on the IBC dataset, averaged across participants and trials. E. Numerical agreement between ground truth
and predicted z-scores. Each dot represents one of the 360 parcels from the HCP parcellation (Glasser et al., 2016),
and z-scores are averaged for all cortical vertices falling within the parcel. Functionally relevant parcels are indicated in
colors, with the marker disambiguating the hemisphere. Higher correlations between true and predicted brain activity
patterns indicate better zero-shot in-silico experiments.
6


---
<!-- Page 7 -->

![Figure 6 (p7, 5979×582px)](tribe_v2_figures/figure_06_p7.png)
![Figure 7 (p7, 7057×5245px)](tribe_v2_figures/figure_07_p7.png)

Figure 5 TRIBE v2 recovers, in-silico, a variety of classic findings in neurolinguistics. TRIBE v2 is tested with various
language functional localizer tasks, extracted from the Individual Brain Charting (IBC) dataset (Pinho et al., 2018).
A. Mean evoked response prediction to ten-word sentences in the "simple" condition of the RSVP task. B. In-silico
experiment categories. These conditions are extracted from the Bang, ArchiSocial, EmotionalPain and RSVP tasks
from IBC. C. Contrast maps obtained from TRIBE’s predictions (see section 5.9). D. Contrast maps computed on the
IBC dataset, averaged across participants and trials. E. Numerical agreement between ground truth and predicted
z-scores. Each dot represents one of the 360 parcels from the HCP parcellation (Glasser et al., 2016), and z-scores are
averaged for all cortical vertices falling within the parcel. Functionally relevant parcels are indicated in colors, with the
marker disambiguating the hemisphere.
7


---
<!-- Page 8 -->

![Figure 8 (p8, 3771×1157px)](tribe_v2_figures/figure_08_p8.png)

Figure 6 Independent component analysis (ICA) shows that TRIBE v2 learns neuroscientifically relevant patterns. A. Top.
TRIBE v2’s ICA. Bottom semantic maps for five neuroscientific keywords generated by Neurosynth (Kent et al.,
2026). In both cases, the 10% vertices with the highest values are shown. B. Spatial correlation (R) between each ICA
component and the functional networks to well-known functional networks.
2.6
In-silico experiments: language
Next, we replicate language experiments from the IBC dataset. For this, we convert the textual stimuli to
an audio file from which we extract word timings, and feed both the text and audio embeddings to TRIBE
v2. As shown in figure 5A, TRIBE v2 yields the expected response to short (10-word) sentences: the elicited
response begins at t=3 seconds in the primary auditory cortex then propagates to the whole language network.
We begin with two localizers of the language network, which contrasts speech segments to either non-speech
segments of a movie or natural sounds (figure 5B). This selects the core language processing areas (figure 5C-E),
in particular associative auditory cortices (A5) and to a lesser degree the superior temporal sulcus (STS) and
Broca’s area (45).
Next, we test the ability of TRIBE v2 to isolate emotional processing regions by contrasting sentences relative
to emotional versus physical pain. TRIBE v2 correctly recovers two important regions: the temporo-parietal
junction (TPJ) and the middle temporal gyrus (MTG). Finally, we replicate two common linguistic contrasts:
sentences versus word lists and complex versus simple sentences. In these two settings, TRIBE v2 correctly
predicts a higher lateralization to the left hemisphere, as well as higher responses in semantic regions (e.g.
TPJ) for the former and syntactic regions (e.g. Broca) for the latter.
Overall, the striking qualitative agreement between the predictions of TRIBE v2 and the outcomes of
classic experiments from the literature demonstrate that beyond its encoding performance in naturalistic
datasets, TRIBE v2 holds promise for the actual downstream task which matters for neuroscience: in silico
experimentation (Jain et al., 2024; Gifford et al., 2025).
2.7
TRIBE v2 learns interpretable representations
To what extent does TRIBE v2 learn interpretable representations of the brain? To investigate this question,
we apply an Independent Component Analysis (ICA) to the final layer of the model, which maps the latent
space of the model to the cortical space (details in section 5.10). Strikingly, the top five components closely
resemble five well-studied function networks in neuroscience (figure 6A): the primary auditory cortex, the
language network, the motion detection area, the default mode network and the visual system (Damoiseaux
et al., 2006). This is confirmed by comparison with cortical maps obtained via NeuroSynth, a meta-analysis
tool which obtains functional maps from common keywords in the neuroscientific literature (Kent et al., 2026):
each component has a high spatial correlation with a single of these functional networks (figure 6B).
2.8
Insights into multimodality
To what extent do the three modalities combined by TRIBE v2 contribute to encoding performance? We
address this question in figure 7A, by assessing the encoding performance of TRIBE v2 retrained with various
8


---
<!-- Page 9 -->

![Figure 9 (p9, 8156×4793px)](tribe_v2_figures/figure_09_p9.png)

Figure 7 TRIBE v2 reveals insights into the multimodality of the human brain. A. Average cortical encoding scores of TRIBE
v2 when trained on subsets of the three modalities for the Algonauts dataset. Error bars denote SEM across the four
participants. B. Highest scoring modality across the whole cortex, with red, green and blue depicting text, audio and
video respectively. C. Difference between the encoding score of TRIBE v2 and the best of its unimodal counterparts. D.
Each vertex of the cortical and subcortical sufaces are color-coded using an RGB mapping where red, green and blue
intensities are determined by the subject-averaged encoding score of the text, audio and video unimodal versions of
TRIBE v2 on the Algonauts dataset. Red, blue and green areas correspond to unimodal areas well encoded by text,
audio and video respectively, while magenta, yellow and cyan correspond to bimodal areas well encoded by text+video,
text+audio and video+audio respectively. Saturation is normalized to make colors more readable. E. Same as D for
each of the four subjects in the Algonauts dataset.
modalities ablated. When training on a single modality, TRIBE v2 achieves significantly lower encoding scores:
text achieves the lowest average encoding score overall, followed by audio, then video. These modalities encode
complementary parts of the brain (figure 7B): as expected, audio dominates near the auditory cortex, video
dominates in the occipital and partietal cortices, while text, which presumably contains the most semantic
information, dominates in the language processing cortices but also in large parts of the prefrontal lobe.
In which areas does multimodality yield the strongest gains? In figure 7C, we compare for each parcel the
encoding score of the multimodal encoder with that of the best of the three unimodal encoders. We observe
that the largest gains are around the temporal-parietal-occipital junction (with up to 50% increase in encoding
score), and to a lesser degree in the prefrontal cortex. These results demonstrate that our multimodal encoder
effectively captures interactions between modalities, which improves whole-brain encoding.
To achieve a more precise understanding of multisensory integration, we overlay the contribution of the three
modalities using an RGB encoding where red, green and blue respectively represent the encoding scores
achieved solely with text, audio and video (figure 7D). The mixing between colors quantifies multisensory
integration: for example, text+audio (yellow) can be observed in the superior temporal lobe and parts of
the ventricles, and video+audio (cyan) can be observed in the ventral and dorsal visual cortices, as well as
in the hippocampus. Note that different subjects display subtle but meaningful differences in the spatial
organization of the multimodality (figure 7E).
Overall, these observations not only align qualitatively with what one would expect from a neuroscientific
9


---
<!-- Page 10 -->

point of view (e.g. Stein and Stanford (2008); Driver and Noesselt (2008); Gao et al. (2023); Tang et al.
(2023a); Hu and Mohsenzadeh (2025)), but also yield valuable insights on how multisensory integration may
occur in the human cortex.
3
## Discussion
The present results strengthen the possibility of a paradigm shift in neuroscience (Hamilton and Huth,
2020), moving from the fragmented mapping of isolated cognitive tasks toward the use of unified, predictive
foundation models of brain and cognitive functions (Mensch et al., 2021; Schneider et al., 2023; Azabou
et al., 2023; Caro et al., 2023; Wang et al., 2025; Binz et al., 2025). By aligning the representations of AI
systems to those of the human brain, we demonstrate that a single architecture can integrate a vast range
of fMRI responses across hundreds of individuals, extending the framework that led the 2025 Algonauts
competition (d’Ascoli et al., 2026). The observed log-linear scaling of encoding accuracy—mirroring power
laws in both artificial intelligence (Kaplan et al., 2020) and neuroscience (Antonello et al., 2023; Beliy et al.,
2024; Banville et al., 2025; d’Ascoli et al., 2025) – suggests that the ceiling for predicting human brain activity
is yet to be reached. Unlike traditional alignment methods such as hyperalignment (Haxby et al., 2011),
which require participants to be exposed to shared stimuli, TRIBE v2’s ability to generalize to novel subjects
and unseen protocols (figure 3) establishes an in silico platform for pre-screening neuroimaging protocols
and augmenting the statistical power of existing datasets. Similarly, and unlike recent decoding models that
reconstruct words (Tang et al., 2023b; Défossez et al., 2023) or images from brain activity (Ozcelik and
VanRullen, 2023; Benchetrit et al., 2023; Scotti et al., 2024), TRIBE v2 serves as a general-purpose encoding
model of whole-brain activity across cognitive domains.
Despite its predictive breadth, TRIBE v2 remains constrained by the inherent spatio-temporal resolution
of fMRI, which cannot capture the millisecond dynamics of neuronal firing. Furthermore, while the model
recovers several canonical neural responses — including selectivity for places and faces (Kanwisher and Yovel,
2006; Chang and Tsao, 2017) as well as linguistic functions (Pallier et al., 2011; Dehaene and Cohen, 2011;
Friederici et al., 2017; Fedorenko et al., 2024; Bhaya-Grossman and Chang, 2022) — its current inputs are
limited to visual, auditory, and semantic features, and thus omit primary sensory modalities such as olfaction,
balance, somatosensation, and their integration (Hedger et al., 2025). A more fundamental limitation is
that the model currently treats the brain as a passive observer of naturalistic stimuli; it does not yet model
the brain as an active agent producing behavior. Integrating neuro-developmental trajectories and clinical
pathology remains a primary goal to move beyond a static, adult brain state and capture the full diversity of
the global population (Henrich et al., 2010).
The transition toward foundation models in neuroscience follows a trajectory recently established in structural
biology (Jumper et al., 2020), chemistry (Abed et al., 2024) and other domains of neuroscience (Wang et al.,
2025), where data-driven unification replaces a series of isolated, small-scale observations. While the "black
box" nature of deep neural networks can obscure the exact premises of their predictions (Linardatos et al.,
2020), TRIBE v2 demonstrates that these models can nonetheless decompose cortical functional hierarchies
and anchor them to specific neural representations (Huth et al., 2016; Margulies et al., 2016). As a robust
"digital model" of the human brain capable of recovering the results of both deep-phenotypic datasets like the
Individual Brain Charting (IBC) repository (Pinho et al., 2018) and large-phenotypic ones like the Human
Connectome Project (Van Essen et al., 2013), TRIBE v2 provides a platform for interpreting neural function
through intervention. As these models continue to scale with the accumulation of naturalistic datasets
(Van Essen et al., 2013; Poldrack et al., 2017; Pinho et al., 2018; Hebart et al., 2019; Nastase et al., 2021;
Allen et al., 2022; St-Laurent et al., 2023; LeBel et al., 2023), they will become indispensable tools for testing
existing theories (Poldrack et al., 2017) and identifying the specific experiments most likely to improve our
understanding of brain mechanics.
10


---
<!-- Page 11 -->

4
Acknowledgements
The authors wish to thank Fernanda Ponce and Bertrand Thirion for their help with the IBC data (Pinho
et al., 2018); Pierre-Louis Xech, Elisa Cascardi and Jennifer Pak for their support; Diego Marcos Segura and
Dominic Giardini for the demo webpage; Valentin Wyart for his advices, and the rest of the Brain and AI
team at Meta AI for insightful discussions.
The Courtois project on neural modelling was made possible by a generous donation from the Courtois
foundation, administered by the Fondation Institut Gériatrie Montréal at CIUSSS du Centre-Sud-de-l’île-de-
Montréal and University of Montreal. The Courtois NeuroMod team is based at “Centre de Recherche de
l’Institut Universitaire de Gériatrie de Montréal”, with several other institutions involved. See the cneuromod
documentation for an up-to-date list of contributors (https://docs.cneuromod.ca).
Data from the Human Connectome Project was provided by the WU-Minn Consortium (Principal Investigators:
David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that
support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience
at Washington University.
11


---
<!-- Page 12 -->

## References
Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison,
Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu,
Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Xin Wang,
Rachel Ward, Yue Wu, Dingli Yu, Cyril Zhang, and Yi Zhang. Phi-4 Technical Report. arXiv, December 2024. doi:
10.48550/arXiv.2412.08905.
Jehad Abed, Jiheon Kim, Muhammed Shuaibi, Brook Wander, Boris Duijf, Suhas Mahesh, Hyeonseok Lee, Vahe
Gharakhanyan, Sjoerd Hoogland, Erdem Irtem, et al. Open catalyst experiments 2024 (ocx24): bridging experiments
and computational models. arXiv preprint arXiv:2411.11783, 2024.
Hossein Adeli, Sun Minni, and Nikolaus Kriegeskorte. Predicting brain activity using transformers. bioRxiv, 2023. doi:
10.1101/2023.08.02.551743. https://www.biorxiv.org/content/early/2023/08/05/2023.08.02.551743.
Sarah Aliko, Jiawen Huang, Florin Gheorghiu, Stefanie Meliss, and Jeremy I Skipper. A naturalistic neuroimaging
database for understanding the brain using ecological stimuli. Scientific data, 7(1):347, 2020.
Emily J Allen, Ghislain St-Yves, Yihan Wu, Jesse L Breedlove, Jacob S Prince, Logan T Dowdle, Matthias Nau, Brad
Caron, Franco Pestilli, Ian Charest, et al. A massive 7t fmri dataset to bridge cognitive neuroscience and artificial
intelligence. Nature neuroscience, 25(1):116–126, 2022.
Richard Antonello, Aditya Vaidya, and Alexander Huth. Scaling laws for language encoding models in fmri. Advances
in Neural Information Processing Systems, 36:21895–21907, 2023.
Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire
Roberts, Koustuv Sinha, Artem Zholus, et al.
V-jepa 2: Self-supervised video models enable understanding,
prediction and planning. arXiv preprint arXiv:2506.09985, 2025.
Mehdi Azabou, Vinam Arora, Venkataramana Ganesh, Ximeng Mao, Santosh Nachimuthu, Michael Mendelson, Blake
Richards, Matthew Perich, Guillaume Lajoie, and Eva Dyer. A unified, scalable framework for neural population
decoding. Advances in Neural Information Processing Systems, 36:44937–44956, 2023.
Hubert Banville, Yohann Benchetrit, Stéphane d’Ascoli, Jérémy Rapin, and Jean-Rémi King. Scaling laws for decoding
images from brain activity. arXiv preprint arXiv:2501.15322, 2025.
Michael S Beauchamp. See me, hear me, touch me: multisensory integration in lateral occipital-temporal cortex.
Current opinion in neurobiology, 15(2):145–153, 2005.
Roman Beliy, Navve Wasserman, Amit Zalcher, and Michal Irani. The wisdom of a crowd of brains: A universal brain
encoder. arXiv preprint arXiv:2406.12179, 2024.
Yohann Benchetrit, Hubert Banville, and Jean-Rémi King. Brain decoding: toward real-time reconstruction of visual
perception. arXiv preprint arXiv:2310.19812, 2023.
Ilina Bhaya-Grossman and Edward F Chang. Speech computations of the human superior temporal gyrus. Annual
review of psychology, 73:79–102, 2022.
Marcel Binz, Elif Akata, Matthias Bethge, Franziska Brändle, Fred Callaway, Julian Coda-Forno, Peter Dayan, Can
Demircan, Maria K. Eckstein, Noémi Éltető, Thomas L. Griffiths, Susanne Haridi, Akshay K. Jagadish, Li Ji-An,
Alexander Kipnis, Sreejan Kumar, Tobias Ludwig, Marvin Mathony, Marcelo Mattar, Alireza Modirshanechi,
Surabhi S. Nath, Joshua C. Peterson, Milena Rmus, Evan M. Russek, Tankred Saanum, Johannes A. Schubert,
Luca M. Schulze Buschoff, Nishad Singhi, Xin Sui, Mirko Thalmann, Fabian J. Theis, Vuong Truong, Vishaal
Udandarao, Konstantinos Voudouris, Robert Wilson, Kristin Witte, Shuchen Wu, Dirk U. Wulff, Huadong Xiong,
and Eric Schulz. A foundation model to predict and capture human cognition. Nature, 644(8078):1002–1009, 2025.
doi: 10.1038/s41586-025-09215-4.
Josue Ortega Caro, Antonio H de O Fonseca, Christopher Averill, Syed A Rizvi, Matteo Rosati, James L Cross, Prateek
Mittal, Emanuele Zappala, Daniel Levine, Rahul M Dhodapkar, et al. Brainlm: A foundation model for brain
activity recordings. BioRxiv, pages 2023–09, 2023.
Charlotte Caucheteux and Jean-Rémi King. Brains and algorithms partially converge in natural language processing.
Communications biology, 5(1):134, 2022.
Le Chang and Doris Y Tsao. The code for facial identity in the primate brain. Cell, 169(6):1013–1028, 2017.
12


---
<!-- Page 13 -->

Omar Chehab, Alexandre Defossez, Jean-Christophe Loiseau, Alexandre Gramfort, and Jean-Remi King. Deep
recurrent encoder: A scalable end-to-end network to model brain signals. arXiv preprint arXiv:2103.02339, 2021.
Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, and Yonghui Wu. W2v-bert:
Combining contrastive learning and masked language modeling for self-supervised speech pre-training. In 2021
IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pages 244–250. IEEE, 2021.
Simon Dahan, Gabriel Bénédict, Logan Zane John Williams, Yourong Guo, Daniel Rueckert, Robert Leech, and
Emma Claire Robinson. Sim: Surface-based fmri analysis for inter-subject multimodal decoding from movie-watching
experiments. In The Thirteenth International Conference on Learning Representations, 2025.
Anders M Dale. Optimal experimental design for event-related fmri. Human brain mapping, 8(2-3):109–114, 1999.
Jessica S Damoiseaux, Serge ARB Rombouts, Frederik Barkhof, Philip Scheltens, Cornelis J Stam, Stephen M Smith,
and Christian F Beckmann. Consistent resting-state networks across healthy subjects. Proceedings of the national
academy of sciences, 103(37):13848–13853, 2006.
Stéphane d’Ascoli, Jérémy Rapin, Yohann Benchetrit, Hubert Banville, and Jean-Rémi King. Tribe: Trimodal
brain encoder for whole-brain fmri response prediction. In The Fourteenth International Conference on Learning
Representations, 2026.
Alexandre Défossez, Charlotte Caucheteux, Jérémy Rapin, Ori Kabeli, and Jean-Rémi King.
Decoding speech
perception from non-invasive brain recordings. Nature Machine Intelligence, 5(10):1097–1107, 2023.
Stanislas Dehaene and Laurent Cohen. The unique role of the visual word form area in reading. Trends in cognitive
sciences, 15(6):254–262, 2011.
Rahul S Desikan, Florent Ségonne, Bruce Fischl, Brian T Quinn, Bradford C Dickerson, Deborah Blacker, Randy L
Buckner, Anders M Dale, R Paul Maguire, Bradley T Hyman, et al. An automated labeling system for subdividing
the human cerebral cortex on mri scans into gyral based regions of interest. Neuroimage, 31(3):968–980, 2006.
Adrien Doerig, Tim C Kietzmann, Emily Allen, Yihan Wu, Thomas Naselaris, Kendrick Kay, and Ian Charest.
Semantic scene descriptions as an objective of human vision. arXiv preprint arXiv:2209.11737, 10, 2022.
Dota Tianai Dong and Mariya Toneva. Vision-language integration in multimodal video transformers (partially) aligns
with the brain. arXiv preprint arXiv:2311.07766, 2023.
Jon Driver and Toemme Noesselt. Multisensory interplay reveals crossmodal influences on ‘sensory-specific’brain
regions, neural responses, and judgments. Neuron, 57(1):11–23, 2008.
Stéphane d’Ascoli, Corentin Bel, Jérémy Rapin, Hubert Banville, Yohann Benchetrit, Christophe Pallier, and Jean-Rémi
King. Towards decoding individual words from non-invasive brain recordings. Nature Communications, 16(1):10521,
2025.
Semih Eren, Deniz Kucukahmetler, and Nico Scherf. Multimodal recurrent ensembles for predicting brain responses to
naturalistic movies (algonauts 2025). arXiv preprint arXiv:2507.17897, 2025.
Oscar Esteban, Christopher J Markiewicz, Ross W Blair, Craig A Moodie, A Ilkay Isik, Asier Erramuzpe, James D
Kent, Mathias Goncalves, Elizabeth DuPre, Madeleine Snyder, et al. fmriprep: a robust preprocessing pipeline for
functional mri. Nature methods, 16(1):111–116, 2019.
Evelina Fedorenko, Anna A Ivanova, and Tamar I Regev. The language network as a natural kind within the broader
landscape of the human brain. Nature Reviews Neuroscience, 25(5):289–312, 2024.
Jean A Frazier, Sufen Chiu, Janis L Breeze, Nikos Makris, Nicholas Lange, David N Kennedy, Martha R Herbert,
Eileen K Bent, Vamsi K Koneru, Megan E Dieterich, et al. Structural brain magnetic resonance imaging of limbic
and thalamic volumes in pediatric bipolar disorder. American Journal of Psychiatry, 162(7):1256–1265, 2005.
Angela D Friederici, Noam Chomsky, Robert C Berwick, Andrea Moro, and Johan J Bolhuis. Language, mind and
brain. Nature human behaviour, 1(10):713–722, 2017.
Chuanji Gao, Jessica J Green, Xuan Yang, Sewon Oh, Jongwan Kim, and Svetlana V Shinkareva. Audiovisual
integration in the human brain: a coordinate-based meta-analysis. Cerebral Cortex, 33(9):5574–5584, 2023.
Alessandro T Gifford, Domenic Bersch, Marie St-Laurent, Basile Pinsard, Julie Boyle, Lune Bellec, Aude Oliva,
Gemma Roig, and Radoslaw M Cichy. The algonauts project 2025 challenge: How the human brain makes sense of
multimodal movies. arXiv preprint arXiv:2501.00504, 2024.
13


---
<!-- Page 14 -->

Alessandro T Gifford, Maya A Jastrzębowska, Johannes JD Singer, and Radoslaw M Cichy. In silico discovery of
representational relationships across visual cortex. Nature Human Behaviour, 9(10):2079–2098, 2025.
Bruno L Giordano, Michele Esposito, Giancarlo Valente, and Elia Formisano. Intermediate acoustic-to-semantic
representations link behavioral and neural responses to natural sounds. Nature Neuroscience, 26(4):664–672, 2023.
Matthew F Glasser, Timothy S Coalson, Emma C Robinson, Carl D Hacker, John Harwell, Essa Yacoub, Kamil
Ugurbil, Jesper Andersson, Christian F Beckmann, Mark Jenkinson, et al. A multi-modal parcellation of human
cerebral cortex. Nature, 536(7615):171–178, 2016.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al.
The llama 3 herd of models.
arXiv preprint
arXiv:2407.21783, 2024.
Umut Güçlü and Marcel AJ Van Gerven. Modeling the dynamics of human brain activity with recurrent neural
networks. Frontiers in computational neuroscience, 11:7, 2017.
Liberty S Hamilton and Alexander G Huth. The revolution will not be controlled: natural stimuli in speech neuroscience.
Language, cognition and neuroscience, 35(5):573–582, 2020.
Charles R Harris, K Jarrod Millman, Stéfan J Van Der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric
Wieser, Julian Taylor, Sebastian Berg, Nathaniel J Smith, et al. Array programming with numpy. Nature, 585
(7825):357–362, 2020.
James V Haxby, J Swaroop Guntupalli, Andrew C Connolly, Yaroslav O Halchenko, Bryan R Conroy, M Ida Gobbini,
Michael Hanke, and Peter J Ramadge. A common, high-dimensional model of the representational space in human
ventral temporal cortex. Neuron, 72(2):404–416, 2011.
Martin N Hebart, Adam H Dickter, Alexis Kidder, Wan Y Kwok, Anna Corriveau, Caitlin Van Wicklin, and Chris I
Baker. Things: A database of 1,854 object concepts and more than 26,000 naturalistic object images. PloS one, 14
(10):e0223792, 2019.
Nicholas Hedger, Thomas Naselaris, Kendrick Kay, and Tomas Knapen. Vicarious body maps bridge vision and touch
in the human brain. Nature, 650(8100):173, 2025.
Joseph Henrich, Steven J Heine, and Ara Norenzayan. The weirdest people in the world? Behavioral and brain sciences,
33(2-3):61–83, 2010.
Yu Hu and Yalda Mohsenzadeh. Neural processing of naturalistic audiovisual events in space and time. Communications
Biology, 8(1):110, 2025.
Alexander G Huth, Wendy A De Heer, Thomas L Griffiths, Frédéric E Theunissen, and Jack L Gallant. Natural speech
reveals the semantic maps that tile human cerebral cortex. Nature, 532(7600):453–458, 2016.
Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda
Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, Andrew Zisserman,
Oriol Vinyals, and Jo¯ao Carreira. Perceiver IO: A General Architecture for Structured Inputs & Outputs. arXiv,
July 2021. doi: 10.48550/arXiv.2107.14795.
Shailee Jain, Vy A Vo, Leila Wehbe, and Alexander G Huth. Computational language modeling and the promise of in
silico experimentation. Neurobiology of Language, 5(1):80–106, 2024.
Shashank Mohan Jain. Hugging face. In Introduction to transformers for NLP: With the hugging face library and
models to solve problems, pages 51–67. Springer, 2022.
Mark Jenkinson, Christian F Beckmann, Timothy EJ Behrens, Mark W Woolrich, and Stephen M Smith.
Fsl.
Neuroimage, 62(2):782–790, 2012.
John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Kathryn Tunyasuvunakool, Olaf
Ronneberger, Russ Bates, Augustin Žídek, Alex Bridgland, et al. Alphafold 2. Fourteenth Critical Assessment of
Techniques for Protein Structure Prediction, 13, 2020.
Nancy Kanwisher and Galit Yovel. The fusiform face area: a cortical region specialized for the perception of faces.
Philosophical Transactions of the Royal Society B: Biological Sciences, 361(1476):2109–2128, 2006.
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec
Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,
2020.
14


---
<!-- Page 15 -->

Kendrick N Kay, Thomas Naselaris, Ryan J Prenger, and Jack L Gallant. Identifying natural images from human
brain activity. Nature, 452(7185):352–355, 2008.
Alexander JE Kell, Daniel LK Yamins, Erica N Shook, Sam V Norman-Haignere, and Josh H McDermott.
A
task-optimized neural network replicates human auditory behavior, predicts brain responses, and reveals a cortical
processing hierarchy. Neuron, 98(3):630–644, 2018.
James D Kent, Nicholas Lee, Angela R Laird, Taylor Salo, Julio Peraza, Katherine L Bottenhorn, Kendra Oudyk,
Thomas E Nichols, Jean-Baptiste Poline, and Alejandro de la Vega. Neurosynth compose: A web-based platform for
flexible and reproducible neuroimaging meta-analysis. Imaging Neuroscience, 4:IMAG–a, 2026.
Qingcheng Kong et al. Toward a human-like visual system: Decoding and calibrating high-level semantic representations
from brain activity. arXiv preprint arXiv:2402.12345, 2024.
Nikolaus Kriegeskorte. Deep neural networks: a new framework for modeling biological vision and brain information
processing. Annual review of vision science, 1:417–446, 2015.
Benjamin Lahner, Kshitij Dwivedi, Polina Iamshchinina, Monika Graumann, Alex Lascelles, Gemma Roig, Alessan-
dro Thomas Gifford, Bowen Pan, SouYoung Jin, N Apurva Ratan Murty, et al. Modeling short visual events through
the bold moments video fmri dataset and metadata. Nature communications, 15(1):6241, 2024.
Benjamin Lahner, Mayukh Deb, N Apurva Ratan Murty, and Aude Oliva. Mosaic: A scalable framework for fmri
dataset aggregation and modeling of human vision. bioRxiv, pages 2025–11, 2025.
Amanda LeBel, Lauren Wagner, Shailee Jain, Aneesh Adhikari-Desai, Bhavin Gupta, Allyson Morgenthal, Jerry Tang,
Lixiang Xu, and Alexander G Huth. A natural language fmri dataset for voxelwise encoding models. Scientific
Data, 10(1):555, 2023.
Jixing Li, Shohini Bhattasali, Shulin Zhang, Berta Franzluebbers, Wen-Ming Luh, R Nathan Spreng, Jonathan R
Brennan, Yiming Yang, Christophe Pallier, and John Hale. Le petit prince multilingual naturalistic fmri corpus.
Scientific data, 9(1):530, 2022.
Pantelis Linardatos, Vasilis Papastefanopoulos, and Sotiris Kotsiantis. Explainable ai: A review of machine learning
interpretability methods. Entropy, 23(1):18, 2020.
Drew Linsley, Pinyuan Feng, and Thomas Serre. Can deep neural networks learn biological vision? arXiv preprint
arXiv:2504.16940, 2025.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
Nikos Makris, Jill M Goldstein, David Kennedy, Steven M Hodge, Verne S Caviness, Stephen V Faraone, Ming T Tsuang,
and Larry J Seidman. Decreased volume of left and total anterior insular lobule in schizophrenia. Schizophrenia
research, 83(2-3):155–171, 2006.
Daniel S Margulies, Satrajit S Ghosh, Alexandros Goulas, Marcel Falkiewicz, Julia M Huntenburg, Georg Langs, Gleb
Bezgin, Simon B Eickhoff, F Xavier Castellanos, Michael Petrides, et al. Situating the default-mode network along
a principal gradient of macroscale cortical organization. Proceedings of the National Academy of Sciences, 113(44):
12574–12579, 2016.
Mackenzie Weygandt Mathis, Adriana Perez Rotondo, Edward F Chang, Andreas S Tolias, and Alexander Mathis.
Decoding the brain: From neural representations to mechanistic models. Cell, 187(21):5814–5832, 2024.
Arthur Mensch, Julien Mairal, Bertrand Thirion, and Gaël Varoquaux. Extracting representations of cognition across
neuroimaging studies improves brain decoding. PLoS computational biology, 17(5):e1008795, 2021.
Juliette Millet, Charlotte Caucheteux, Yves Boubenec, Alexandre Gramfort, Ewan Dunbar, Christophe Pallier, Jean-
Remi King, et al. Toward a realistic model of speech processing in the brain with self-supervised learning. Advances
in Neural Information Processing Systems, 35:33428–33443, 2022.
Tom M Mitchell, Svetlana V Shinkareva, Andrew Carlson, Kai-Min Chang, Vicente L Malave, Robert A Mason, and
Marcel Adam Just. Predicting human brain activity associated with the meanings of nouns. science, 320(5880):
1191–1195, 2008.
Samuel A Nastase, Yun-Fei Liu, Hanna Hillman, Asieh Zadbood, Liat Hasenfratz, Neggin Keshavarzian, Janice Chen,
Christopher J Honey, Yaara Yeshurun, Mor Regev, et al. The “narratives” fmri dataset for evaluating models of
naturalistic language comprehension. Scientific data, 8(1):250, 2021.
15


---
<!-- Page 16 -->

Xuan-Bac Nguyen, Xudong Liu, Xin Li, and Khoa Luu. The algonauts project 2023 challenge: Uark-ualbany team
solution. arXiv preprint arXiv:2308.00262, 2023.
Subba Reddy Oota, Jashn Arora, Vijay Rowtula, Manish Gupta, and Raju S Bapi. Visio-linguistic brain encoding.
arXiv preprint arXiv:2204.08261, 2022.
Furkan Ozcelik and Rufin VanRullen. Natural scene reconstruction from fmri signals using generative latent diffusion.
Scientific Reports, 13(1):15666, 2023.
Christophe Pallier, Anne-Dominique Devauchelle, and Stanislas Dehaene. Cortical representation of the constituent
structure of sentences. Proceedings of the National Academy of Sciences, 108(6):2522–2527, 2011.
Ana Luísa Pinho, Alexis Amadon, Torsten Ruest, Murielle Fabre, Elvis Dohmatob, Isabelle Denghien, Chantal Ginisty,
Séverine Becuwe-Desmidt, Séverine Roger, Laurence Laurier, et al. Individual brain charting, a high-resolution fmri
dataset for cognitive mapping. Scientific data, 5(1):180105, 2018.
Russell A Poldrack, Chris I Baker, Joke Durnez, Krzysztof J Gorgolewski, Paul M Matthews, Marcus R Munafò,
Thomas E Nichols, Jean-Baptiste Poline, Edward Vul, and Tal Yarkoni. Scanning the horizon: towards transparent
and reproducible neuroimaging research. Nature reviews neuroscience, 18(2):115–126, 2017.
Blake A Richards, Timothy P Lillicrap, Philippe Beaudoin, Yoshua Bengio, Rafal Bogacz, Amelia Christensen, Claudia
Clopath, Rui Ponte Costa, Archy de Berker, Surya Ganguli, et al. A deep learning framework for neuroscience.
Nature neuroscience, 22(11):1761–1770, 2019.
Christina Sartzetaki, Gemma Roig, Cees GM Snoek, and Iris IA Groen. One hundred neural networks and brains
watching videos: Lessons from alignment. bioRxiv, pages 2024–12, 2024.
Daniel Carlström Schad, Shrey Dixit, Janis Keck, Viktor Studenyak, Aleksandr Shpilevoi, and Andrej Bicanski. Vibe:
Video-input brain encoder for fmri response modeling. arXiv preprint arXiv:2507.17958, 2025.
Steffen Schneider, Jin Hwa Lee, and Mackenzie Weygandt Mathis. Learnable latent embeddings for joint behavioural
and neural analysis. Nature, 617(7960):360–368, 2023.
Martin Schrimpf, Jonas Kubilius, Ha Hong, Najib J Majaj, Rishi Rajalingham, Elias B Issa, Kohitij Kar, Pouya
Bashivan, Jonathan Prescott-Roy, Franziska Geiger, et al. Brain-score: Which artificial neural network for object
recognition is most brain-like? BioRxiv, page 407007, 2018.
Paul S Scotti, Mihir Tripathy, Cesar Kadir Torrico Villanueva, Reese Kneeland, Tong Chen, Ashutosh Narang, Charan
Santhirasegaran, Jonathan Xu, Thomas Naselaris, Kenneth A Norman, et al. Mindeye2: Shared-subject models
enable fmri-to-image with 1 hour of data. arXiv preprint arXiv:2403.11207, 2024.
Michael N Shadlen and William T Newsome. Neural basis of a perceptual decision in the parietal cortex (area lip) of
the rhesus monkey. Journal of neurophysiology, 86(4):1916–1936, 2001.
Siddharth Srivastava and Gaurav Sharma. Omnivec2 - a novel transformer based network for large scale multimodal
and multitask learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
(CVPR), pages 27412–27424, June 2024.
Marie St-Laurent, Basile Pinsard, Oliver Contier, Katja Seeliger, Valentina Borghesani, Julie Boyle, Pierre Bellec,
and Martin Hebart. cneuromod-things: a large-scale fmri dataset for task-and data-driven assessment of object
representation and visual memory recognition in the human brain. Journal of Vision, 23(9):5424–5424, 2023.
Barry E Stein and Terrence R Stanford. Multisensory integration: current issues from the perspective of the single
neuron. Nature reviews neuroscience, 9(4):255–266, 2008.
Jerry Tang, Meng Du, Vy Vo, Vasudev Lal, and Alexander Huth. Brain encoding models based on multimodal
transformers can transfer across language and vision. Advances in neural information processing systems, 36:
29654–29666, 2023a.
Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G Huth. Semantic reconstruction of continuous language
from non-invasive brain recordings. In Nature Neuroscience, volume 26, pages 858–866, 2023b.
Mariya Toneva and Leila Wehbe. Interpreting and improving natural-language processing (in machines) with natural
language-processing (in the brain). Advances in neural information processing systems, 32, 2019.
David C Van Essen, Stephen M Smith, Deanna M Barch, Timothy EJ Behrens, Essa Yacoub, Kamil Ugurbil, Wu-
Minn HCP Consortium, et al. The wu-minn human connectome project: an overview. Neuroimage, 80:62–79,
2013.
16


---
<!-- Page 17 -->

Marcel Van Gerven and Sander Bohte. Artificial neural networks as models of neural information processing, 2017.
Nishitha Vattikonda, Aditya R Vaidya, Richard J Antonello, and Alexander G Huth. Brainwavlm: Fine-tuning speech
representations with brain responses to language. arXiv preprint arXiv:2502.08866, 2025.
Cesar Kadir Torrico Villanueva, Jiaxin Cindy Tu, Mihir Tripathy, Connor Lane, Rishab Iyer, and Paul S Scotti.
Predicting brain responses to natural movies with multimodal llms. arXiv preprint arXiv:2507.19956, 2025.
Brian A Wandell and Jonathan Winawer. Imaging retinotopic maps in the human brain. Vision research, 51(7):
718–737, 2011.
Aria Y Wang, Kendrick Kay, Thomas Naselaris, Michael J Tarr, and Leila Wehbe. Incorporating natural language into
vision models improves prediction and understanding of higher visual cortex. BioRxiv, pages 2022–09, 2022.
Eric Y Wang, Paul G Fahey, Zhuokun Ding, Stelios Papadopoulos, Kayla Ponder, Marissa A Weis, Andersen Chang,
Taliah Muhammad, Saumil Patel, Zhiwei Ding, et al. Foundation model of neural activity predicts response to new
stimulus types. Nature, 640(8058):470–477, 2025.
Haiguang Wen, Junxing Shi, Yizhen Zhang, Kun-Han Lu, Jiayue Cao, and Zhongming Liu. Neural encoding and
decoding with deep learning for dynamic natural vision. Cerebral cortex, 28(12):4136–4160, 2018.
Weihao Xia, Raoul de Charette, Cengiz Oztireli, and Jing-Hao Xue. Umbrae: Unified multimodal brain decoding. In
European Conference on Computer Vision, pages 242–259. Springer, 2024.
Daniel LK Yamins and James J DiCarlo. Using goal-driven deep learning models to understand sensory cortex. Nature
neuroscience, 19(3):356–365, 2016.
Daniel LK Yamins, Ha Hong, Charles F Cadieu, Ethan A Solomon, Darren Seibert, and James J DiCarlo. Performance-
optimized hierarchical models predict neural responses in higher visual cortex. Proceedings of the national academy
of sciences, 111(23):8619–8624, 2014.
Huzheng Yang, James Gee, and Jianbo Shi. Memory encoding model. arXiv preprint arXiv:2308.01175, 2023.
Yanjun Zhou et al. Clip-decoding: Toward human-level visual decoding through augmented training. arXiv preprint
arXiv:2401.09918, 2024.
17


---
<!-- Page 18 -->

5
## Methods
5.1
Approach overview
Our objective is to predict the brain activity of participants exposed to naturalistic stimuli. This is framed as
a high-dimensional regression task where the targets are the blood-oxygen-level-dependent (BOLD) signals
detected by fMRI recording devices. We predict both at the cortical level, where the targets are the 20,484
vertices of the fsaverage5 surface (Jenkinson et al., 2012), and at the subcortical level, where the targets are
the 8,802 voxels of 8 subcortical regions defined by the Harvard-Oxford atlas (Frazier et al., 2005; Makris
et al., 2006; Desikan et al., 2006).
Our model takes as input three stimulus modalities (or as many as available, depending on the study): (i)
the video clip being viewed by the participant, (ii) the audio being heard and (iii) the transcript of what is
being heard. From these, we extract high-dimensional embeddings from the (frozen) intermediate layers of
state-of-the art generative AI models along three modalities of interest: text, audio and video.
This provides a timeseries of multimodal embeddings, which we feed to a (trainable) transformer to aggregate
information across time. Each timestep is then projected to a low-dimensional latent space, from which a
subject-conditioned layer predicts the brain responses at each selected area.
5.2
Feature extraction
In this section, we describe our pipeline for feature extraction, which relies on modality-specific pretrained AI
models. Note that these models are frozen, ensuring robustness to out-of-distribution data, as demonstrated
by the results of TRIBE v1 on the Algonauts 2025 competition (d’Ascoli et al., 2026).
Text embeddings
We extract "timed" text embeddings from the timestamped transcripts of the videos. For
each word w to embed, we prepend the preceding k = 1, 024 words in the transcript, which we feed through
Llama-3.2-3B (Grattafiori et al., 2024). For each intermediate layer l, we extract the token(s) overlapping
with the word w and average them to obtain a contextualized word embedding of dimension Dtext = 2048.
We then construct an evenly spaced grid at a frequency fstim = 2 Hz, and for each time-bin, we sum the
embeddings of words which overlap with the bin. This allows to temporally align the text features with the
audio and video features.
Audio embeddings
To obtain audio embeddings, we extract audio files from the videos, split them into
60-second chunks, then feed these through Wav2Vec-Bert-2.0 (Chung et al., 2021). We then resample the
hidden representations of the latter from 50 Hz to fstim = 2 Hz. For each intermediate layer l, this yields time
series of embeddings of dimension Daudio = 1, 024. Note that the resulting embeddings carry bidirectional
information about both the past and future of the stimulus window, whereas text and video embeddings only
contain information about the past.
Video embeddings
For video embeddings, we again construct an evenly spaced grid at a frequency fstim = 2 Hz,
and for each bin of time, we feed 64 frames spanning the preceding 4 seconds to Video-JEPA-2-Giant (Assran
et al., 2025). For each intermediate layer l, we compress the tensor of activations by averaging over all patch
tokens, yielding a time series of embeddings of size Dvideo = 1, 280. Note that this spatial averaging step was
necessary to keep the size of the tensor manageable. However, it comes at the cost of discarding positional
information, which we expect to deteriorate encoding performance in low-level visual areas which exhibit a
retinotopic mapping (Wandell and Winawer, 2011).
Combining the modalities
For each of the three modalities m, the feature extraction described above leads to
a time series of embeddings at fstim = 2 Hz, with embeddings of shape [Lm, Dm], where Lm and Dm are the
number of layers and dimensionality of the transformer of modality m. To compress these embeddings while
retaining both low-level and high-level information, for each modality, we split the layers into L groups, then
average the tensor per group along the layer dimension, compressing to a shape [L, Dm]. We then concatenate
the layers and feed the resulting vector through a linear layer with a shared output dimension D = 384
18


---
<!-- Page 19 -->

followed by layer normalization. Finally, we concatenate the three modalities, leading to a time series of
multimodal embeddings of shape Dmodel = 3 × 384 = 1152. This will be the input to our transformer encoder.
Implementation details
We extract stimuli features from pretrained language, audio and video models
available on the HuggingFace platform (Jain, 2022) and cache them as Numpy memmap arrays (Harris et al.,
2020) for fast loading during the training of our encoding model. Feature extraction is completed in 24 hours
on 128 V100 GPUs with 32GB of VRAM, and model training lasts 24 hours on a single such GPU.
5.3
Model
Transformer encoder
We extract windows of duration T = 100 seconds from these embedding time series,
add learnable positional embeddings and a learnable subject embedding, then feed the result through a
Transformer encoder1 with 8 layers and 8 attention heads. This enables information to be exchanged between
timesteps. At the output of the transformer, we use an adaptive average pooling layer to decimate the
embeddings from the stimulus frequency fstim = 2 Hz to the resampled fMRI frequency ffMRI = 1 Hz.
Modality dropout
One desirable property of a multimodal encoding model is its ability to provide meaningful
predictions in the absence of one or several modalities, for example for a silent movie or a podcast. To
encourage this behaviour, while at the same time avoiding excessive reliance on one modality, we introduce
modality dropout: during training, we randomly mask off each modality by zeroing out the corresponding
input tensor with a probability p = 0.3, resampling such that at least one modality is left unmasked.
Subject block
TRIBE uses a subject block to handle subject specificities during training, inspired by (Défossez
et al., 2023). Specifically, we use a subject-conditional linear layer to project the transformer outputs to
the Ntarget encoding targets, which are either the vertices of the fsaverage5 cortical surface or the voxels of
the subcortical regions. Denoting the number of subjects as S, the subject block can be viewed as a tensor
of shape (S, Dmodel, Ntargets) whose first dimension is indexed by the subject. This subject block can be
fine-tuned on new subjects, as detailed in section 5.4.
Unseen subject prediction
For in-silico experimentation, one is generally interested in the group response to
a particular stimulus. For this, we implement a form of "subject dropout": during training, with probability
p = 0.1, we bypass the subject block and instead feed the latents through a special "unseen subject" linear
layer. This forces the model to maintain accurate predictions even without information on the subject, and
enables our model to be used to predict group responses in a zero-shot manner.
5.4
Training and validation
Training
TRIBE is penalized via a mean-squared error term between the predicted and ground truth
fMRI data, without any additional regularisation.
We train it for up to 15 epochs with the AdamW
optimizer (Loshchilov and Hutter, 2017) using a batch size of 16. The learning rate is warmed up linearly to
10−4 over the first 10% of steps, then decayed following a cosine learning rate schedule. We use early stopping
based on the validation Pearson score, with a patience of 3 epochs. Model training can be performed under a
day on a single V100 GPU with 32GB of VRAM thanks to the efficiency of the cached feature extraction,
detailed in section 5.2
Validation
To avoid any form of data leakage, we ensure that the stimuli in the validation set have no overlap
with the stimuli in the training set. For this, we either use the predefined splits defined by the authors, or
define our own splits by randomly assigning 10% of the podcasts/stories/movies to be held out. Our encoding
metric is the Pearson correlation R. To compute it, we collect, for each subject and each parcel, the predicted
and ground truth fMRI responses across all TRs of the validation set and compute the Pearson correlation.
We then average the values over all subjects and parcels. We also refer to this metric as the “encoding score”
of the model.
1We use the transformer implementation from the x-transformers package available at https://github.com/lucidrains/
x-transformers.
19


---
<!-- Page 20 -->

Fine-tuning
Finetuning is then performed for a single epoch, with all parameters of TRIBE unfrozen, using
the same hyperparameter configuration as described for training. Note that for fine-tuning, the subject block
is retrained with N test
subj ̸= N train
subj
given by the OOD study. To initialize the weights for the new subject block,
the simplest option would be to use Lavg for each test subject. However, since the number of subjects in the
OOD studies is an order of magnitude larger than the training studies, the subject block obtained in this way
(of shape (S, Dmodel, Ntargets), with Ntargets = 20, 484) can become prohibitively large. To circumvent this,
we use low-rank matrix factorization: we factorize Lavg as a product of low-rank matrices i.e. Lavg ≃USV T ,
with a bottleneck rank r = 128, using the SVD algorithm provided by torch.svd. The matrix SV T is then
parametrized by a normal linear layer, while the matrix U, of shape (r, Ntargets) becomes the new subject
block.
5.5
Linear baseline
To provide a baseline for TRIBE, we built a linear multimodal brain encoder by simply replacing the
transformer encoder by a linear convolution layer whose kernel spans 9 TRs and is offset by 5 seconds relative
to the stimuli. The latter serves as a temporal receptive field to aggregate the signal across time. Note that
this model is technically a deep linear model, since it contains a sequence of three linear operations: (i) the
linear projection from the modality-specific embeddings to the shared mutimodal embedding space, (ii) the
learnable convolutional layer and (iii) the final projection to cortical space. This linear model is trained in the
same conditions as TRIBE, but with a maximum of 30 epochs instead of 15, as we observed that convergence
is slower. We experiment both training this model across subjects, like TRIBE, and also training it separately
on each subject. Interestingly, even this linear model benefits from multi-subject pretraining (figure 2).
5.6
Statistics
To assess which cortical vertices are predicted above chance in figure 2, we follow Huth et al. (2016) and
compare the estimated correlations to the null distribution of correlations between two independent Gaussian
random vectors of the same length. Given the autocorrelation inherent to fMRI signals, we only keep one TR
every 60 seconds to ensure the independence of the samples.
In figure 2 and figure 3, we use paired t-tests with FDR correction across subjects to compare performances
of different models.
5.7
Datasets
In this section, we present the eight datasets that we curated from the literature for training and testing of
our model: see table 1 for a summary of their key features:
• Training: Our model is trained jointly on four "deep" fMRI datasets where a small amount of participants
were exposed to a large volume of naturalistic stimuli in varying conditions: silent videos (Wen et al.,
2018) (3 subjects, 35 hours), speech (LeBel et al., 2023) (8 subjects, 85 hours), videos with sound but
no speech (Lahner et al., 2024) (10 subjects, 61 hours) and multimodal videos (Gifford et al., 2024) (4
subjects, 265 hours).
• Testing: Our model is tested on four "broad" fMRI datasets where a large amount of participants were
exposed to a small volume of naturalistic stimuli, two during speech listening Nastase et al. (2021); Li
et al. (2022) (433 subjects, 326 hours) and two during movie watching Aliko et al. (2020); Van Essen
et al. (2013) (262 subjects, 338 hours). The choice to leave these for testing is motivated both by the
observation that training on deep datasets leads to better performance (Antonello et al., 2023), and by
the fact that many subjects are required to obtain reliable estimates of the group-averaged response.
Courtois NeuroMod (Gifford et al., 2024)
The Courtois NeuroMod dataset consists of 3T recordings of six
healthy human participants who watched the same naturalistic videos, namely the first six seasons of the
popular TV series Friends as well as four movies: The Bourne Supremacy, Hidden Figures, The Wolf of Wall
Street and Life (a BBC Nature documentary). This amounts to an unprecedently large recording volume of
over 80 hours of fMRI per subject. In the present work, we focus on a subset of four subjects curated for the
20


---
<!-- Page 21 -->

Algonauts 2025 competition (Gifford et al., 2024) (the other two subjects are not publicly available at the
time of writing). We use the fMRI data preprocessed by the authors using fMRIPrep (Esteban et al., 2019),
co-registered to the MNI152NLin2009cAsym template.
Lebel2023 (LeBel et al., 2023)
This dataset was recorded on a 3T machine while 8 healthy participants each
listened to 27 complete, natural, narrative stories (370 minutes) from The Moth podcast over the course of
five scanning sessions. Three of these participants also listened to a further 57 complete stories (629 minutes).
We preprocess the data using fMRIPrep, co-registering to the MNI152NLin2009cAsym template.
BoldMoments (Lahner et al., 2024)
This dataset was recorded on a 3T machine while 10 healthy participants
watched 1,102 3-second video clips sampled from the Memento10k dataset. The clips were manually selected
to encompass videos that contained movement (i.e., not static content), were filmed in a natural context,
and represented a wide selection of possible events a human might witness. We use the train and test splits
provided by the authors (train videos are viewed 3 times while test videos are viewed 10 times). We use the
fMRI data preprocessed by the authors, co-registered to the MNI152NLin2009cAsym template.
Wen2017 (Wen et al., 2018)
This dataset was recorded on a 3T machine while 3 healthy participants watched
three hours of short video clips without sound. The clips were collected from Youtube and VideoBlocks and
concatenated together into 8-minute continuous streams. Importantly, this is the only dataset considered
without any audio content.
We use the fMRI data preprocessed by the authors, co-registered to the
MNI152NLin6Asym template.
Naturalistic NeuroImaging Database (Aliko et al., 2020)
This dataset was recorded on a 3T machine while
86 healthy participants watched one of ten full-length movies. We use the fMRI data preprocessed by the
authors, which is co-registered to the Colin27 template.
Human Connectome Project (Van Essen et al., 2013)
This dataset was recorded on a 7T machine as part
of the Human Connectome Project, while 167 healthy participants watched a full movie. This dataset was
selected for its large amount of participants and the 7T resolution, leading to a higher signal to noise ratio.
We use the fMRI data preprocessed by the authors, co-registered to the MNI152NLin6Asym template.
5.8
fMRI preprocessing
The datasets employed in this study feature diverse preprocessing pipelines and formats. Ideally, all datasets
would be processed through an identical fMRIPrep workflow to yield time series in consistent volumetric and
surface spaces (e.g., MNI152NLin2009cAsym and fsaverage). However, heterogeneities in data organization
(e.g., non-BIDS compliance) and missing metadata rendered this approach unfeasible. We therefore adopted a
harmonized workflow designed to minimize preprocessing variance and enhance model generalizability.
With the exception of the Lebel2023 dataset, all BOLD data were provided by the original authors already
registered to a standard volumetric template (MNI152NLin2009cAsym, MNI152NLin6Asym, or Colin27).
For the Lebel2023 dataset, we performed preprocessing using fMRIPrep with default parameters, targeting
the MNI152NLin2009cAsym template.
Cortical extraction
To map these volumetric data onto the fsaverage surface, we utilized the following
procedure. For each template T, we computed the geometric coordinates of the fsaverage white and pial
surfaces within the coordinate space of T using FreeSurfer’s recon-all and surf-to-surf tools. We then projected
the volumetric time series onto these surfaces using nilearn.surface.vol_to_surf. Specifically, we employed
the “ball” sampling method with a 3 mm radius centered at a depth halfway between the pial and white matter
boundaries.
Subcortical extraction
To extract subcortical signals from the BOLD images, we mask them with the Harvard-
Oxford atlas (Frazier et al., 2005; Makris et al., 2006; Desikan et al., 2006), defined in MNI space at a
resolution of 2 millimeters. This results in 8,802 voxels spanning 8 subcortical regions: hippocampus, lateral
ventricles, amygdala, thalamus, caudate, putamen, pallidum and accumbens.
21


---
<!-- Page 22 -->

Rescaling and detrending
The timeseries for each vertex were z-scored across each session. Importantly,
detrending was then applied to the timeseries. This step was not included in the preprocessing of the Algonauts
2025 competition: in our experiments, neglecting this step led to confounds on datasets where slow drifts are
particularly salient, such as Wen2017 and Lebel2023. Indeed, due to the long context window of our model,
these drifts could be exploited by the encoding model to spuriously increase its encoding score in otherwise
hard to predict brain areas. For this reason, detrending makes the encoding task performance significantly
lower, but better reflects downstream encoding performance.
Resampling
Finally, the signals are linearly resampled to ffMRI = 1 Hz using np.interpolate in order for
the frequency to be consistent across the datasets, which use different repetition times.
Hemodynamic lag
There is a roughly 5-second delay between stimuli presentation and peak response in the
brain. For this reason, we offset the fMRI timeseries by 5 seconds relative to the stimuli, meaning that to
predict a window [0,T] of fMRI, TRIBE takes as input a window [-5, T-5] of stimuli.
5.9
In-silico experiments
In this section, we describe our protocol for replicating the IBC tasks in-silico. Note that in all these
experiments, TRIBE is used in unseen-subject mode (see section 5.3).
Visual experiments
We replicate the visual experiments from the FaceBody and Visu tasks of IBC, whose
images are available at https://github.com/individual-brain-charting/public_protocols. All images are
presented to TRIBE in a randomized order, for one second every eight seconds (transforming them to static
videos). Note that the face image in figure 4 is here used solely for illustrative purposes. The actual stimuli
used for the study are available at https://github.com/individual-brain-charting/public_protocols/tree/
master/FaceBody/stimuli.
Language experiments
We replicate the experiments Bang, Audio, EmotionalPain and RSVP from the IBC
datasets. For Bang, since the original movie was not available for download, we simply contrast segments
from the Algonauts dataset which contain speech versus those which do not. For Audio, we display the audio
segments in a randomized order, with 8 seconds between each segment. For EmotionalPain and RSVP, where
the stimuli consist in sentences, the replication is more involved since we need to obtain word timings. For
this, we first translate the sentences from French to English, then feed the texts through a text-to-speech
pipeline2 which yields an audio file. We then run a speech-to-text pipeline3 to obtain word timings, which
provides the expected dataframe of multimodal (text+audio) events.
Contrast maps
To obtain contrast maps for the visual experiments, we obtain contrast maps by simply
selecting the predicted response at t=5 after the image is shown (which is the peak of the response as shown
in figure 4A), and substracting the average responses at t=5 for the other categories.
To obtain contrast maps for the lanugage experiments, we follow the protocol used for the IBC dataset.
Specifically, we fit a General Linear Model (GLM), where the regressors are estimated through the convolution
of TRIBE’s predicted BOLD responses with the canonical Hemodynamic Response Function. We use the
implementation provided by nilearn’s FirstLevelModel with default parameters.
Regions of interest
To extract regions of interest, we used the Glasser Multimodal parcellation (Glasser
et al., 2016). FFA, EBA, PPA, VWFA, A5, 45, STS, TPJ, MTG respectively correspond to the following ROI
labels: FFC, V4t, PH, A5, 45, STSv, PGi, TE1a.
2https://github.com/kyutai-labs/pocket-tts
3https://github.com/m-bain/whisperX
22


---
<!-- Page 23 -->

5.10
Independent component analysis
For this analysis, we use the "unseen" subject layer of TRIBE, a tensor of shape (Dmodel, Ntargets) which maps
from the latent space to each subject’s cortical space. To visualize the latent space, we apply Independent
Component Analysis on this matrix, using the FastICA implementation provided by sklearn with default
parameters and n_components = 5. Note that at odds with Principal Component Analysis, the order of the
components does not carry any meaning for ICA. The result is a collection of five vectors of size Ntargets
which we can plot on the cortical surface.
To characterize these brain maps functionally, we compare them with five functional maps obtained by
NeuroSynth (Kent et al., 2026) with the following keywords: "primary auditory", "language", "motion",
"default network" and "visual". These maps are resampled from volumetric to cortical space via nilearn’s
vol_to_surf, then compared with the ICA components via Pearson correlation across all vertices of the
cortical surface.
23


---
<!-- Page 24 -->

## Appendix
A
Related works
Limitations of current encoding models
Motivated by the alignment between the representations learnt by
AI models and those observed in the human brain (Huth et al., 2016; Schrimpf et al., 2018; Caucheteux and
King, 2022), several teams have built encoding models to predict brain responses to natural stimuli from
the activations of neural networks in response to images (Yang et al., 2023; Adeli et al., 2023; Nguyen et al.,
2023), speech (Millet et al., 2022) and text (Toneva and Wehbe, 2019). However, these encoding models are
currently limited in three critical ways.
First, linearity: existing encoding approaches typically rely on ridge regression to map the AI model
representations onto those of the brain. This assumes that these two sets of representations are linearly
related, a phenomenon known not to be true (Linsley et al., 2025). Second, specificity: due to large variability
in brain responses across subjects, tasks and brain areas, existing encoding approaches typically train separate
models for each subject, task and brain area, which prevents them from learning the co-occurring patterns
which emerge in large heterogeneous datasets. Third, unimodality: most existing encoding approaches predict
brain responses from unimodal stimuli, which makes them incapable of capturing how the brain integrates
information from multiple modalities (Hu and Mohsenzadeh, 2025). This is particularly limiting as it has
been shown that cross-modal interactions occur not only in specific multisensory areas (Gao et al., 2023;
Beauchamp, 2005), but also in primary sensory areas (Driver and Noesselt, 2008; Stein and Stanford, 2008).
Deep encoders
While there has been significant research on deep learning for multimodal brain decoding (Da-
han et al., 2025; Scotti et al., 2024; Xia et al., 2024; Zhou et al., 2024; Kong et al., 2024), the corresponding
literature for brain encoding is more sparse. However, some recent works suggest to train recurrent models to
predict brain responses from frozen visual or linguistic features (Güçlü and Van Gerven, 2017; Chehab et al.,
2021), or fine-tune existing pretrained models using the brain encoding objective Vattikonda et al. (2025).
Closest to our approach is Beliy et al. (2024), which uses a transformer-based approach to predict the visual
cortex. While these works relax the linearity assumption, they are restricted to a single sensory modality.
Multimodal encoders
Conversely, a few studies have built encoding models on top of vision-language
transformers, demonstrating gains compared to unimodal transformers (Dong and Toneva, 2023; Oota et al.,
2022; Doerig et al., 2022; Wang et al., 2022; Tang et al., 2023a). However, these works rely solely on linear
mappings to model brain responses from the activations of the multimodal transformers. We believe this can be
suboptimal for two reasons. First, multimodal transformers are still relatively new: at rare exceptions (Jaegle
et al., 2021; Srivastava and Sharma, 2024; Abdin et al., 2024), they often only integrate static images and
text (audio and video being significantly more compute-intensive), and tend to lag behind the performance of
unimodal transformers. Second, and more fundamentally, the way these models integrate information across
modalities may be very different from how the human brain does such multimodal integration. An ideal
encoding pipeline should thus learn how to best combine different modalities.
Comparison to TRIBE v1
Motivated by the Algonauts 2025 brain encoding competition Gifford et al. (2024),
many teams have introduced multimodal brain encoding models, with some notable entries listed in table 2.
Among these, an earlier iteration of our current model, TRIBE v1, achieved the state-of-the-art. Compared to
TRIBE v1, our model addresses the aforementioned shortcomings: (i) it is tasked to predict the 20k vertices
of the fsaverage5 cortical map as well as subcortical regions (versus 1000 parcel activations from a cortical
atlas for TRIBE v1); (ii) it is trained on 25 participants performing four different tasks (versus 4 participants
on one task for TRIBE v1); (iii) it is equipped with an "unseen subject" module, which enables it to be tested
on 695 participants coming from four held-out studies (TRIBE v1 was not tested on unseen subjects), (iv)
thanks to the variety of tasks it was exposed to during training, TRIBE v2 generalizes to the non-naturalistic
stimuli such as the controlled images of the IBC protocol figure 4 and enables in-silico experimentation (for
which TRIBE v1 was not tested).
24


---
<!-- Page 25 -->

![Figure 10 (p25, 2784×947px)](tribe_v2_figures/figure_10_p25.png)

Rank
Reference
Mean score
Subject 1
Subject 2
Subject 3
Subject 5
1
Ours
0.2146 ± 0.0312
0.2381
0.2105
0.2377
0.1720
2
(Schad et al., 2025)
0.2096 ± 0.0283
0.2353
0.2046
0.2268
0.1718
3
(Eren et al., 2025)
0.2094 ± 0.0215
0.2233
0.2072
0.2271
0.1798
4
(Villanueva et al., 2025)
0.2085 ± 0.0267
0.2295
0.2003
0.2300
0.1743
5
Unpublished
0.2055 ± 0.0291
0.2306
0.2010
0.2240
0.1662
Table 2 TRIBE v2 achieves the first place in the Algonauts 2025 brain prediction competition (Gifford et al 2024). We report the
mean score ± standard deviation for the top five out of 263 teams.
Auditory
Multisensory
Visual
(Early)
(Early)
Visual
(Ventral)
Visual
(Dorsal)
Visual
(MT Complex)
(TPJ)
Auditory
(Association)
Inferior
Frontal
BoldMoments
Courtois NeuroMod
Lebel2023
Wen2017
0.5
0.4
0.3
Pearson
0.2
0.1
RI
44
45
V1
V2
V3
V4
V6
V7
A4
A5
A1
V4t
TA2
PSL
PEF
IFJa
IFJp
LO1
LO2
LO3
STV
V3B
V3A
V6A
IFSa
IFSp
IPS1
LBelt
PBelt
V3CD
PHA1
PHA2
PHA3
VMV1
VMV2
VMV3
MBelt
TPOJ3
TPOJ2
TPOJ1
STSva
STSvp
STSda
STSdp
ROI
Figure 8 Performance of TRIBE across cortical regions.
25

**Table 1 (page 25)**

|  | isua Early Bold | l ) Moments |  |  |  | Visual (Ventral) Courto |  |  | Visual (Dorsal) is NeuroMod Lebel2023 |  |  |  |  |  |  |  |  |  | Visual (MT Complex) Wen2017 |  |  |  |  |  | Multisensory (TPJ) |  |  |  |  |  |  | Auditory (Association) |  |  |  |  |  |  |  | Auditory (Early) |  |  |  |  |  |  | Inferior Frontal |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |



---
<!-- Page 26 -->

![Figure 11 (p26, 2165×2435px)](tribe_v2_figures/figure_11_p26.png)

Figure 9 Examples of decoded BOLD timeseries for various areas of the brain.
26


---
<!-- Page 27 -->

![Figure 12 (p27, 2066×1902px)](tribe_v2_figures/figure_12_p27.png)

Figure 10
Examples of video segments and the corresponding predicted and true brain responses.
In all cases, the
brain activity is normalized to the 99% percentile across the segment. See more examples in our interactive demo
https://aidemos.atmeta.com/tribev2/.
27