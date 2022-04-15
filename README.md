Implemantation of K-MEANS algorithm and Code description
In Phase 1 I implemented basic functions to read data from csv files and generate histograms for different attributes. Also, I wrote function to calculate mean, median, standard deviation and variance for each attribute. Following functions were implemented in Phase 1.
main
Read csv file and store into dataframe.
Fill ‘na’ values using ffill method.
Iterate over each column except from ‘A2’ to ‘A10’ and print mean, median, std and var.
Generate histograms for all attributes with bin size 10.
In Phase 2 I implemented k-means algorithm. Following functions were implemented in phase 2.
A. initial
It takes data as input.
Get random row numbers to select.
Prints selected index and data row at that position.
Returns both rows and mu2 and mu4.
B. assign
It takes data and two random rows we got from initialization step as input.
These two rows are center for two different classes for now.
This function will iterate over each row and calculate Euclidean distance between two points and based on that it will put it in bucket of two different classes.
At the end it will return these buckets (lists)
C. recompute
It takes data, class-2 index list and class-4 index list as input.
It returns new means for class-2 and class-4.
D. main
Main function was modified to call initial function first and get random rows.
Iterate less than 50 times and call assignment function to get new class buckets or until previous and current assignments match. If it doesn’t match do recalculation of mean.
Print total iterations, final centroid values and first 20 rows showing actual and predicted class.
In Phase 3 I implemented error function to calculate error rate for each class and total error rate as well.
