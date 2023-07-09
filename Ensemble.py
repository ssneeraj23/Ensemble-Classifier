import pandas as pd
import numpy as np
import random 
from sklearn.svm import SVC 
import matplotlib.pyplot as plt
import math
from sklearn import metrics
from sklearn.neural_network import MLPClassifier

def ss_norm(df):
    df_std = df.copy()
    for column in df_std.columns:
        if column=="label" : break
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()   
    return df_std
   

def train_test_split_manual(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df 

def forward_select(pd_data):
 
    instance_id_colname = pd_data.columns[0]
    no_of_columns = len(pd_data.columns) # number of columns
    class_column_index = no_of_columns - 1
    class_column_colname = pd_data.columns[class_column_index]
    no_of_available_attributes = no_of_columns - 2
    available_attributes_df = pd_data.drop(columns = [
        instance_id_colname, class_column_colname]) 
    optimal_attributes_df = pd_data[[instance_id_colname,class_column_colname]]
    base_performance = -9999.0
    while no_of_available_attributes > 0: 
        best_performance = -9999.0
        best_attribute = "Placeholder"
        for col in range(0, len(available_attributes_df.columns)):
            this_attr = available_attributes_df.columns[col]
            temp_opt_attr_df = optimal_attributes_df.copy()
            temp_opt_attr_df.insert(
                loc=1,column=this_attr,value=(available_attributes_df[this_attr])) 
            # Run the best mlp model from part 3 on this new dataframe and return the 
            # classification accuracy            
            train_df,test_df=train_test_split_manual(temp_opt_attr_df,0.2)
            X=train_df.values
            Y=test_df.values
            X_labl=X[:,-1]
            X_data=X[:,:-1]
            Y_labl=Y[:,-1]
            Y_data=Y[:,:-1]
            if accuracy_1>=accuracy_2:  
                mlp_model_1 = MLPClassifier( max_iter = 5000, hidden_layer_sizes=(16,),solver='sgd',learning_rate_init=0.001,batch_size=32)
            else: mlp_model_1 = MLPClassifier( max_iter = 5000, hidden_layer_sizes=(256,16),solver='sgd',learning_rate_init=0.001,batch_size=32)
            mlp_model_1.fit(X_data,X_labl)
            current_performance=mlp_model_1.score(Y_data,Y_labl)
            # Find the new attribute that yielded the greatest
            # classification accuracy
            if current_performance > best_performance:
                best_performance = current_performance
                best_attribute = this_attr
        # Did adding another feature lead to improvement?
        if best_performance > base_performance:
            base_performance = best_performance
            # Add the best attribute to the optimal attribute data frame
            optimal_attributes_df.insert(
                loc=1,column=best_attribute,value=(
                available_attributes_df[best_attribute]))
            # Remove the best attribute from the available attribute data frame
            available_attributes_df = available_attributes_df.drop(
                columns = [best_attribute]) 
            no_of_available_attributes -= 1     
        else:
            break
    return optimal_attributes_df

df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",header=None)
df=df.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,0]]
df.set_axis( ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315','Proline','label'], axis='columns', inplace=True)
df=ss_norm(df)
train_df,test_df=train_test_split_manual(df,0.2)
X=train_df.values
Y=test_df.values
X_labl=X[:,-1]
X_data=X[:,:-1]
Y_labl=Y[:,-1]
Y_data=Y[:,:-1]

f=open("result.txt","w")
f.close()
f=open("result.txt","a")

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_data, X_labl)
accuracy_linear = svm_model_linear.score(Y_data, Y_labl)
print("Accuracy for binary SVM classifier with Linear Kernel is : \n",accuracy_linear)
f.write("Accuracy for binary SVM classifier with Linear Kernel is : \n")
f.write(str(accuracy_linear))
svm_model_rbf = SVC(kernel = 'rbf', C = 1).fit(X_data, X_labl)
accuracy_rbf = svm_model_rbf.score(Y_data, Y_labl)
f.write("\nAccuracy for binary SVM classifier with Radial Basis Function as Kernel is : \n")
f.write(str(accuracy_rbf))
print("Accuracy for binary SVM classifier with Radial Basis Function as Kernel is : \n",accuracy_rbf)
svm_model_quad = SVC(kernel = 'poly',degree=2, C = 2).fit(X_data, X_labl)
accuracy_quad = svm_model_quad.score(Y_data, Y_labl)
print("Accuracy for binary SVM classifier with Quadratic as Kernel is : \n",accuracy_quad)
f.write("\nAccuracy for binary SVM classifier with Quadratic as Kernel is : \n")
f.write(str(accuracy_quad)) 

mlp_model_1 = MLPClassifier( max_iter = 5000, hidden_layer_sizes=(16,),solver='sgd',learning_rate_init=0.001,batch_size=32)
mlp_model_1.fit(X_data,X_labl)
accuracy_1=mlp_model_1.score(Y_data,Y_labl)
print("\nAccuracy for MLP classifier using with stochastic gradient descent optimiser and learning rate as 0.001 and batch size of 32 are as follows")
f.write("\n\nAccuracy for MLP classifier using with stochastic gradient descent optimiser and learning rate as 0.001 and batch size of 32 are as follows \n")
print("With 1 hidden layer with 16 nodes : ",accuracy_1)
f.write("With 1 hidden layer with 16 nodes : ")
f.write(str(accuracy_1)+"\n")
mlp_model_2 = MLPClassifier( max_iter=5000, hidden_layer_sizes=(256,16,), solver='sgd', learning_rate_init=0.001,batch_size=32)
mlp_model_2.fit(X_data,X_labl)
accuracy_2=mlp_model_2.score(Y_data,Y_labl)
print("\nWith  2 hidden layers with 256 and 16 nodes respectively : ",accuracy_2)
print("")
f.write("With  2 hidden layers with 256 and 16 nodes respectively : ")
f.write(str(accuracy_2)+"\n")

if accuracy_1>=accuracy_2:
    hls=(16,)
else : hls=(256,16,) 

acc_list=[]
l_rate=[0.1, 0.01, 0.001, 0.0001,0.00001]
for lr in l_rate:
    model_temp = MLPClassifier( max_iter = 5000, hidden_layer_sizes=hls,solver='sgd',learning_rate_init=lr,batch_size=32)
    model_temp.fit(X_data,X_labl)
    accuracy_temp=mlp_model_1.score(Y_data,Y_labl)
    acc_list.append(accuracy_temp)

x=np.array(l_rate)
y=np.array(acc_list)
f.write("")
f.write("\nThe best accuracy model in part 3 is with ")
if accuracy_1>=accuracy_2: f.write("1 hidden layer with 16 nodes\n")
else :f.write("2 hidden layers with 256 and 16 nodes respectively.\n")
f.write("The accuracies with above MLP classifier for learning rates 0.1, 0.01, 0.001, 0.0001 and 0.00001 are as follows\n")
f.write(str(acc_list[0])+"\n"+str(acc_list[1])+"\n"+str(acc_list[2])+"\n"+str(acc_list[3])+"\n"+str(acc_list[4])+"\n")
plt.plot(x, y)
plt.xlabel('Learning-Rate')
plt.ylabel('Accuracy')
plt.title("Learning-Rate vs Accuracy")
plt.savefig('accuracy')

best_feat_df=forward_select(df)
best_features=best_feat_df.columns.values
print("The best set of features obtained after using forward selection method on best MLP classifier obtained above are ")
f.write("The best set of features obtained after using forward selection method on best MLP classifier obtained above are \n")
for i in range(len(best_features)-1):
    print(best_features[i])
    f.write(str(best_features[i])+"\n")

svm_quad_pred=svm_model_quad.predict(Y_data)
svm_rbf_pred=svm_model_rbf.predict(Y_data)
if accuracy_1>=accuracy_2:
    mlp_pred=mlp_model_1.predict(Y_data)
else: mlp_pred=mlp_model_2.predict(Y_data)

ensemble_pred=[]

for i in range(len(Y_data)):
    if(svm_quad_pred[i]==svm_rbf_pred[i]):ensemble_pred.append(svm_quad_pred[i])
    elif (svm_quad_pred[i]==mlp_pred[i]) :ensemble_pred.append(svm_quad_pred[i])
    elif (svm_rbf_pred[i]==mlp_pred[i]): ensemble_pred.append(svm_rbf_pred[i])
    else :ensemble_pred.append(svm_quad_pred[i])

ensemble_accuracy=metrics.accuracy_score(Y_labl,ensemble_pred)
print("Ensemble learning (max voting technique) using SVM with quadratic, SVM with radialbasis function and the best accuracy model from part 3 has the accuracy: \n",ensemble_accuracy)
f.write("\nEnsemble learning (max voting technique) using SVM with quadratic, SVM with radialbasis function and the best accuracy model from part 3 has the accuracy: \n"+str(ensemble_accuracy)+"\n")

plt.show()