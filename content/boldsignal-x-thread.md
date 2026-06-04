# BoldSignal — X/Twitter Thread

---

[1/10]
I built an experiment on top of Meta's TRIBE v2.

It takes a YouTube video and returns predicted
brain activation across 20,000 cortical voxels -
second by second.

I turned that into a "Brain Score." Here's what
I found. 🧵

---

[2/10]
TRIBE v2 is a brain encoding model from Meta FAIR.

It was trained on real fMRI data - people in
scanners watching naturalistic video.

The model learns to predict BOLD signal across
the cortex from video features alone.

I wrapped it into a batch Colab pipeline.

---

[3/10]
From those voxel predictions, I compute a
Brain Score (0-100) across 4 dimensions:

· Default Mode Network activity
· Amygdala + hippocampus circuit
· pSTS, fusiform, rTPJ (social regions)
· V1-V4, A1, insula in the first 15 seconds

The weights and cutoffs are my own - reasonable
starting points, not validated parameters.

---

[4/10]
The Default Mode Network is the brain's
"mind-wandering" system.

The idea: when it quiets down, attention is
being directed outward. When it activates
during narrative content, it may signal
deeper comprehension.

That distinction changes how I score the
same region depending on content type.

---

[5/10]
Each video also gets a second-by-second
timeline of predicted attention and
emotional activation.

Useful for seeing where things change -
a hook that fires and then drops off
looks very different from flat engagement
throughout.

Make of it what you will.

---

[6/10]
Under the hood:

· TRIBE v2 encoder extracts video features
· Predictions mapped to fsaverage5 surface
· ~20,000 voxels per 2-second timestep
· ROI masks via Schaefer 2018 + Destrieux

Same surface space used in published
cognitive neuroscience research.

---

[7/10]
I ran it on 5 YouTube Shorts.

Scores ranged from 59 to 66 - tighter than
I expected. Not sure if that's the format,
my normalisation, or both.

The notebook is open. Run it on your own
content and tell me what you get.

---

[8/10]
The scoring weights are my choices - I
read the literature and made judgment calls.

Someone with more fMRI experience would
probably make different ones.

That's part of why I'm sharing it early.
#OpenSource

---

[9/10]
What this is not:

· Not real fMRI data
· Not validated against actual engagement
· Not a substitute for audience research

The numbers are directional. The interesting
part is the shape - where things rise and
fall across the video, not the final score.

---

[10/10]
Notebook is open. One click to run in Colab.

Drop a URL in Cell 8 and see what comes out.
I'm curious whether the patterns hold across
different content types.

GitHub: https://github.com/deep0patel/boldsignal
Colab: https://colab.research.google.com/github/deep0patel/boldsignal/blob/main/boldsignal_batch.ipynb

#Neuroscience #MachineLearning

---
