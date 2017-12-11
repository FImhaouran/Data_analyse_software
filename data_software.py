import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class Data_values(object):

    def __init__(self, filename):
        self.filename = filename

    def make_dir(self):
        csv_file = 'db/%s' %(self.filename)
        path = 'location'

        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
        return(csv_file)

    def read_data_pandas(self):
        csv_file= self.make_dir()
        df = pd.read_csv(csv_file)
        return(df)

    def shape_data(self):
        df = self.read_data_pandas()
        (rows, columns) = df.shape
        return("---Data blabla--- \nRows: %d \nColumns: %d" % (rows, columns))

    def features(self):
        df = self.read_data_pandas()
        all_features = [i for i in df]
        return("Features: %s" %(all_features))

    # description of the each feauture with the mean, std max and min value
    def describe_data(self):
        df = self.read_data_pandas()
        describe = df.describe()

        # we will make a dictionary that contains all the features that have numeric cells,
        # The different values will be put in their arrays
        # from these features the mean, lowest, highest number and std will be taken.
        boxplot_info = {}

        for column in describe:
            boxplot_info["%s" %(column)] = []

            # count, mean, std, low
            for i in range(1,4):
                # rounded = round(describe[column][i], 3)
                boxplot_info["%s" %(column)].append(describe[column][i])

            # high
            boxplot_info["%s" %(column)].append(describe[column][7])
        # now all the selected values (mean, std low, high are solely selected and everything is in the boxplot_info dictionary)
        return(boxplot_info)

    def plot_description(self):
        # get the dictionary boxplot_info
        features_values = self.describe_data()
        # list all features that are holded by describe :  all columns with numeric values
        features = [key for key, value in features_values.items()]
        amount_features = len(features)

        x_as = np.arange(1,2)
        # Data needed for the plots
        values = [value for key, value in features_values.items()]
        means = [values[i][0] for i in range(len(features_values))]
        std = [values[i][1] for i in range(len(features_values))]
        mins = [values[i][2] for i in range(len(features_values))]
        maxes = [values[i][3] for i in range(len(features_values))]


        for i in range(amount_features):
            print('------------------------------------------------------------------Plotting and Saving -> %s-ErrorPlot\
    ------------------------------------------------------------------'%(features[i]))

            # making the plot
            fig = plt.figure(i+1)
            fig.suptitle('ErrorPlot')
            plt.errorbar(x_as, means[i],[[means[i]-mins[i]], [maxes[i]-means[i]]], ecolor='purple',capsize = 18, capthick =2)
            plt.errorbar(x_as, means[i], std[i], fmt='_', ecolor='lightblue',mec='darkblue',mew = 1,ms =30, mfc = 'darkgreen',capsize = 0, capthick =0.5,lw=30)
            ax = fig.add_subplot(111)
            ax.set_xlabel('%s' %(features[i]))
            ax.set_xticklabels([])
            plt.savefig('location')
        print("\n")

class Data_cleaning(Data_values):
    # def __init__(self, filename):
    #     self.filename = filename
    #
    def make_dir(self):
        csv_file = 'db/%s' %(self.filename)
        path = plt.savefig('location')
        path1 = plt.savefig('location')
        try:
            os.makedirs(path)

        except OSError:
            if not os.path.isdir(path):
                raise
        try:
            os.makedirs(path1)

        except OSError:
            if not os.path.isdir(path1):
                raise
        return(csv_file)

    def missing_values(self):
        csv_file = self.make_dir()
        df = pd.read_csv(csv_file)
        # describes amount of filled in values per feature
        count_rows = df.count()
        # describes the amount of missing values per feature
        amount_missing = df.isnull().sum()
        return(count_rows,amount_missing)

    def missing_values_plot(self):
        count_rows, amount_missing = self.missing_values()
        features_amount = len(count_rows)
        features = []
        filled = []
        missed = []
        x_places = np.arange(features_amount)

        for key,value in count_rows.items():
            features.append(key)
            filled.append(value)


        for key,value in amount_missing.items():
            missed.append(value)

        height = 1.5*max(filled)
        p1 = plt.bar(x_places, filled ,align ='center',width = 0.5, color = 'green')
        p2 = plt.bar(x_places, missed,bottom = filled,align ='center', width = 0.5, color = 'grey')
        plt.xticks(x_places, features,rotation = 40, fontsize = 8.7)
        plt.yticks(np.arange(0,height, (1/6)*height))
        plt.legend((p1[0], p2[0]), ('Filled_values', 'Missing_values'))
        plt.ylabel("Amount")
        plt.xlabel("Features")
        plt.title("Amount of missing data and filled data")
        print('------------------------------------------------------------------Plotting and Saving -> %s-F&M_plots\
------------------------------------------------------------------'%(self.filename))

        plt.savefig('location' %())
    print("\n")

    def fill_missing_values(self):
        csv_file = self.make_dir()
        df = pd.read_csv(csv_file)
        # look at the values with integers and fill the missing values with the mean
        features_values = self.describe_data()
        # list all features that are holded by describe :  all columns with numeric values
        features = [key for key, value in features_values.items()]
        amount_features = len(features)

        # Filling all empty cells of features with integer values (i.e. that resulted from describe)
        for feature in features:
            df.new = df[features].fillna(df[features].mean())

        return(df)

    def plot_2features(self, feature1, feature2):

        df = self.fill_missing_values()

        # all unique variable of the features
        unique_values = {}
        count_variables = {}

        # loop through feature 1 and see how many categories of data input there is
        # put them in dict_features
        loop_features = [feature1,feature2]
        # select only the rows that we use
        df = df[loop_features]

        # loop through the list of features
        for feature in loop_features:
            # make key for each feature
            unique_values['%s'%(feature)] = []
            count_variables['%s'%(feature)] = []
            # loop through feature values and store all the unique values in the key
            for value in df[feature]:
                if value not in unique_values['%s'%(feature)]:
                    unique_values['%s'%(feature)].append(value)

        # We count the number of times each variable occurs
        # grab feature 1 loop through each value and count all the variables that feature2 has, plot
        count_each_var= []
        for i in range(len(unique_values[feature1])):

            # getting list with one value of feature1
            df_one_value1 = df[df[feature1] == unique_values[feature1][i]]
            # count the values that occur in feature2
            count_values = df_one_value1[feature2].value_counts()
            # make total sum
            total = df_one_value1[feature2].value_counts().sum()
            # when plotting , name of the plot is feature1,i
            # print('%s:%d:'%(feature1,i))
            # each part of pie  count_values/total
            print("Feature1:%d\n"%(unique_values[feature1][i]))

            labels = []
            sizes = []
            explode = []
            # loop through key(feature value) and the value of the key(count of feature value)
            for j,p in count_values.items():
                print("Feature2:%d, count:%d"%(j,p))
                labels.append(j)
                a = p/total*100
                sizes.append(a)
                explode.append(0.05)

            # print("Feature1:%d\n"%(unique_values[feature1][i]))

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            # plt.title()
            fig1.suptitle('%s=%d:'%(feature1,unique_values[feature1][i]), fontsize=12, fontweight='bold')
            # fea
            print("--------------------------Plotting pie chart and saving--------------------------------%s=%d:"%(feature1,unique_values[feature1][i]))

            plt.savefig('location' %()))

            print("\n")
                        # Pie chart, where the slices will be ordered and plotted counter-clockwise:

    def string_to_num(self):

        # get the dataframe
        df = self.fill_missing_values()

        # loop through all columns/attributes
        # look at the attribute, if it is numeric contiue else replace each\
        # occuring string(unique_values) with a digit

        # Dictionary with columns that have string unique_values, and dictionary with the same columns but assiging digits
        unique_values = {}
        unique_num = {}
        for column in df:
            # each attribute gets a key in the dictionary "unique_values"
            unique_values[(column)] = []
            unique_num[(column)] = []

            # if values in column is not float or int but string/object, replace the values with number

            if (df[column].dtype)==object:
                for value in df[column]:
                    if value not in unique_values[column]:
                        unique_values[column].append(value)
                # print(unique_values[column])
                for amountValues in range(len(unique_values[column])):
                    unique_num[column].append(amountValues)

                df[column] = df[column].replace(unique_values[column],unique_num[column])

        return(df)


# Filling all missing values in the column with the mean
    def missing(self):
        df = self.string_to_num()
        for column in df:
            missing =
            [column].isnull().sum()
            if missing >= 0:
                df[column] = df[column].fillna(df[column].mean())
        return(df)




# show correlation between all attributes
    def correlation(self):

        df = self.missing()
        corr = df[df.columns].corr()
        pal = sns.light_palette("navy", as_cmap=True)
        sns.heatmap(corr,annot = True, cmap=pal)
        plt.show()


# test different attributed in different machine learning algorithms

def main():
    t1 = Data_values('test.csv')
    p1 = Data_cleaning('train.csv')
    p1.make_dir()

    # print(p1.string_to_num())
    p1.correlation()
    # p1.missing()

if __name__ == '__main__':
    main()
