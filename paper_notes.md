# Paper notes
Notes from the paper.

---
## Feature engineering method 1: selecting 7 features
7 Features listed in paper (terms in brackets are cursive and were written in math format. See paper page 12/13):

1. body length (or "[d_ST]"): distance from snout to base of tail
2. [d_SF]: distance of front paws to base of tail relative to body length (formally: [d_SF] = [d_ST] - [d_FT], where [d_FT] is the distance between front paws and base of tail
3. [d_SB]: distance of back paws to base of tail relative to body length (formally: [d_SB] = [d_ST] - [d_BT]
4. Inter-forepaw distance (or "[d_FP]"): the distance between the two front paws
5. snout speed (or "[v_s]"): the displacement of the snout location over a period of 16ms
6. base-of-tail speed (or ["v_T"]): the displacement of the base of the tail over a period of 16ms
7. snout to base-of-tail change in angle: **(TODO: requires explanation....)**

After, author also specifies that: the features are also smoothed-over, or averaged across, a sliding 
window of size equivalent to 60ms (30ms prior to and after the frame of interest).

The author on clustering features after above feature extraction is explained:
>####Data clustering
>With sampling frequency at 60 Hz, 1 frame
every 16 ms, we are capturing fragments of movements. Any
clustering algorithm will have a difficult time teasing apart
the innate spectral nature of action groups. To resolve this
issue, we decided to either take the sum over all fragments
for time-varying features (features 5-7), or the average across
the static measurements (features 1-4) every 6 frames. Due
to our sliding window smoothing prior to this step at about
double the resolution of the bins, we are not concerned with
washing out inter-bin behavioral signals.

Furthermore,...


