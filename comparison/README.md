# COLMAP Matcher Comparison

Dataset:
AP03 grouped calibrated COLMAP

Exhaustive matcher:
- 268 registered images
- Single sparse model

Sequential matcher:
- Two sparse models
- Model 0: 143 registered images
- Model 1: 124 registered images

Conclusion:
The exhaustive matcher reconstructs the sequence as one connected model,
whereas the sequential matcher splits it into two disconnected components.
