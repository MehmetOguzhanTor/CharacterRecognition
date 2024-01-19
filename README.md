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

