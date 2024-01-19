# CharacterRecognition
In this project, I am working on character recognition on a digit dataset using MATLAB.The digit dataset consists of 5000 rows of handwritten image of digits in the range of {0, 1, 2, 3 ,4 ,5 ,6 ,7 ,8, 9}. The image of each digit is digitized by 20*20 matrix which adds up to 400 – dimensional vectors. The dataset also has labels set which will help training the algorithm and finding accuracy of it in the end. The programming environment for the project is MATLAB. And, in the process of implementing what is asked, some toolboxes and codes are used for the second and third question which will be mentioned in the following parts. Lastly, some background information is learned to perform some of the tasks like classifying a model with Gaussian distribution.

Calculations and Plots

Principal Components Analysis is required to implement to the 400-dimensional data to lower dimensional subspaces.

The first thing that is done is to shuffle the dataset and labels to distribute them randomly. Then, the shuffled dataset and labels are split into two parts as one of them will be the trainset and the other will be the test set. The splitting is done 2500 by 2500 which will add up to 5000. Then, to implement gaussian model into the training data, mean, std and variance of the trainset is found which is followed by centralizing the train set by extracting the mean of it. Then, the covariance array is found by taking the square of centralized array. From the covariance matrix, eigen vectors and eigen values are found together. Last thing for this part is to order the eigen values and eigen vectors in descending order because we will start choosing them from the large ones. In the following graph, eigen values can be observed in descending order.

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/ccd7492d-c4c3-45a5-9eb5-09c0be5cac79)

As it can be observed from the graph, eigen values starts from high values and decreases to zero. Since,
the number of eigen values is 400 which is high and the eigen values do not change significantly.
Specifically, if around 50 margin of eigen values is observed, it can be seen that the decrease is much
smaller than before 50 margin. Normally, just by looking at the plot, it is not possible to determine the
optimal number of eigen values should be taken into consideration. However, just by observing the plot,
around 50 components would be chosen.

When the display of ample mean for the whole training dataset as an image is done, the result is like in
the following figure. Yet, almost nothing is in this display except some small dots of white which are
barely visiable.

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/94e978ef-a1c2-4e6d-8af1-7287219588fa)

Figure 2: The display of the sample mean for the training dataset

Then, eigenvectors of the dataset are displayed in the figure 3. In the display, it can be clearly seen that,
at first some digits are visible. These digits represent the first eigen vector which are the higher ones.
As their values decreases, the visibility of their represented digit is also decreasing.

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/2085a875-5d9f-49a9-8a02-f9f2a2699cf8)

Gaussian classification is implemented to our both training and test data by using transformation matrix that is calculated from the training data. 20 different subspace dimensions are chosen for this part in order to plot and observe the classification error vs the number of components used for each subspace. In the code of this part, gauss train and gauss test functions are created in the end of the code in functions part. In these functions’ gauss train function trains our data algorithm and in gauss test function, accuracy is found. This process is done for both train and test data. In the following figure, there is the plot of classification error vs the number of components for train(blue) and test(orange) data. Instead of creating two plots, to make the observation easier, one plot of two error is chosen.

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/b96310d0-3302-4a6e-ab33-238826f8053f)

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/51628e98-04e4-4a3e-bc6a-e162042207af)

As, it can be seen from the plots, when the number of component increase the accuracy of the model also increses and for 20 components it is around 90% which is considerably good prediction.As expected, the accuracy of the train data is higher than the test dataset’s because we construct the model using the train set. So, model predicts better when the train dataset is used. As the number of components increases the difference between the accuracies of train and test dataset can be observed clearly as in Figure 5.

In the second part of the project, Fisher linear discriminant analysis is practiced to the data for the same purpuse in the first part of the project(PCA). Againg half of the data is set to be the train and other half is set to be the test data which are again the same as in the first part. For LDA, a tool used which is availible in the Mathworks[1]. The LDA function that is used for this part is included in the functions part of the MATLAB code. Then, the display function in the first part is used again to display the bases as images. The display is shown in the following figure.

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/d9b5a176-842f-4030-bfc5-a08a78b3ebe3)

For the display, I was expecting a similar result as I got from the PCA bases. However, In the display of the bases resulted from the LDA function, digits cannot be observed clearly for all the eigen vectors.

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/9d89b8cd-bd38-456e-a0d8-2c48770b4457)

As, it can be observed from the plotting, the accuracy of test data is considerably less then accuracy of train data. As mentioned before, the reason is that we construct the data by using the train data, so when we model the algorithm using the train data, the accuracy of the model increases.

In the third part of the project, articles about Sammon’s mapping and t-SNE are analyzed. These articles are for mapping a dataset to two dimensions. There is a function tool in MATLAB for Sammon’s mapping [4] and for t-SNE there is a direct function in MATLAB [5]. When these functions are obtained according to the project assignment, and when we plot the resulting vectors using scatter function, we get the following 2 figures. In Sammon’s mapping we could use the ‘MaxIter’ option to decrease the run time of the code. However, the more the max iteration is the less the error of the function will be.
As the articles mentions, in the Sammon mapping, there is a topological structure where as in the t-SNE the structure seems like gathered in the middle of the plotting.

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/49e95a3a-8ebc-458e-b923-b6b4e228eb06)

![image](https://github.com/MehmetOguzhanTor/CharacterRecognition/assets/116079107/cb445c59-61cd-435e-a5fd-66a6d6db94e0)

References 1. Mathworks.com. 2021. Linear Discriminant Analysis (LDA) aka. Fisher Discriminant Analysis (FDA). [online] Available at: <https://www.mathworks.com/matlabcentral/fileexchange/53151-linear-discriminant-analysis-lda-aka-fisher-discriminant-analysis-fda> [Accessed 16 April 2021].
2. Sammon’s mapping (J. W. Sammon, “A Nonlinear Mapping for Data Structure Analysis,” IEEE Transactions on Computers, vol. C-18, no. 5, pp:401-409, May 1969)
3. t-SNE (L. J. P. van der Maaten and G. E. Hinton, “Visualizing High-Dimensional Data Using t-SNE,” Journal of Machine Learning Research, vol. 9, pp:2579-2605, November 2008) 
4. MathWorks, www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/56937/versions/28/previews/FSLib_v6.0_2018/lib/drtoolbox/techniques/sammon.m/index.html. 
5. “X.” t, www.mathworks.com/help/stats/tsne.html.
