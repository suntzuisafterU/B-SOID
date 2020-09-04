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
7. snout to base-of-tail change in angle: 

Author also specifies that: the features are also smoothed over, or averaged across, 
    a sliding window of size equivalent to 60ms (30ms prior to and after the frame of interest).


