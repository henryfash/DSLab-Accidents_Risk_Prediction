# Data Science in Practice Lab
Topic: Accidents_Risk_Prediction

Main Data (Source:Kaggle): A car accident dataset, which covers traffic accidents in 49 states of the USA from February 2016 to June 2020. The dataset contains about 3.5 million accident records. https://www.kaggle.com/sobhanmoosavi/us-accidents

For public safety, precautions are necessary to avoid accidents. In 2016, the United States' National Highway Traffic Safety Administration (NHTSA) records show that 37,461 people were killed in 34,436 motor vehicle crashes, an average of 102 per day [1]. Therefore accident risk prediction models can help in saving lives and improving road safety. In our work, we consider prediction as a supervised binary classification task where given spatio-temporal feature information like weather and traffic conditions, we predict probability for "accident" and "non-accident" classes in a specific spatial grid and for a specific time interval which we have taken to be 15 minutes. Besides, in our work we also address some hypothesises. Our work makes use of points of interest on the road like intersections, bridges, etc to improve the prediction accuracy. The accuracy of the deep learning model in particular can be further improved by the addition of similar geographic features, traffic history, etc which can be explored in future works. Our models can be used in roadside assistance functions like finding safer and optimal routes, and also assisting government decisions like finding focal points for improving traffic infrastructure so the budget is spent effectively.

References
1. https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812451
2. Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. "A Countrywide Traffic Accident Dataset.", arXiv preprint
   arXiv:1906.05409 (2019).
