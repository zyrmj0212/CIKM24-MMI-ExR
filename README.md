The author has been juggling several project deadlines recently, as a result, the code associated with the paper will be uploaded at a slightly later date than initially planned.
Please be assured that the author is fully committed to making the code available. In the meantime, if you have any questions or concerns, please do not hesitate to reach out to the author directly via email at zhaoyurou@ruc.edu.cn

# CIKM24-MMI-ExR
Codes and Online Appendix for the CIKM'24 paper "Aligning Explanations for Recommendation with Rating and Feature via Maximizing Mutual Information"
## Dataset
The three datasets are from Amazon (Movie & TV), and
Yelp (2019), and TripAdvisor3 respectively, and their corresponding
recommendation explanation data are collected
from the GitHub repository of (Li, Zhang, and Chen 2021).
Additionally, we utilize the Spacy \footnote{https://spacy.io/} toolkit to conduct sentence dependency analysis on each review, removing those where the noun subject is ``I'' or ``We''. This is because such reviews often lack objective descriptions of the items, making them unsuitable to refer to when generating explanations. Finally, we divide the whole dataset into train/validation/test subsets at a ratio of 8:1:1. The details of the datasets are presented in Table \ref{tab:dataset} .
## Usage
