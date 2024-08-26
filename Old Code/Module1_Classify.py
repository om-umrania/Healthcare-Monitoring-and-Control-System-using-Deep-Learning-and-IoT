import warnings
import classification_functions as cf
import DeepLearningProcess as dlp

warnings.filterwarnings("ignore")

dataset_files = [0] * 3
start_rows = [0] * 3
end_rows = [0] * 3
classes_arr = [0] * 3

#Cancer set
dataset_files[0] = 'Cancer/cancer.csv'
start_rows[0] = 1
end_rows[0] = 23
classes_arr[0] = 24

#Parkinsons
dataset_files[1] = 'Parkinson/Parkinson.csv'
start_rows[1] = 0
end_rows[1] = 7
classes_arr[1] = 8

#Blood Cancer
dataset_files[2] = 'Blood/Blood.csv'
start_rows[2] = 0
end_rows[2] = 14
classes_arr[2] = 15

#Find the values of Di
di_arr = [0]*3
for count in range(0,len(dataset_files)) :
    di_arr2 = [0] * (end_rows[count] - start_rows[count])
    out_count = 0
    for count2 in range(start_rows[count], end_rows[count]) :
        di = cf.findAccuracy(dataset_files[count], count2, count2+1, classes_arr[count])
        di_arr2[out_count] = di
        out_count = out_count + 1
    
    di_arr[count] = di_arr2
    
#Find the Dij array
dij_arr = []
for i in range(0,len(di_arr)) :
    for k in range(0,len(di_arr[i])) :
        for j in range(0,len(di_arr)) :
            dij_vals = [0] * len(di_arr[j])
            for l in range(0,len(di_arr[j])) :
                dij = (di_arr[i][k] + di_arr[j][l])/2
                dij = (dij + di_arr[i][k] + di_arr[j][l]) / 3
                dij_vals[l] = dij
                print('D(%d, %d, %d, %d):%0.04f' % (i,k,j,l,dij))
            dij_arr.extend(dij_vals)
#Now use these as weights, and classify the data
print('***********************************')
print('Processing CANCER Dataset')
dlp.applyDL(dataset_files[0])
acc1 = cf.findAccuracy(dataset_files[0], start_rows[0], end_rows[0], classes_arr[0],1,dij_arr)
print('Processing Parkinsons Dataset')
dlp.applyDL(dataset_files[1])
acc2 = cf.findAccuracy(dataset_files[1], start_rows[1], end_rows[1], classes_arr[1],1,dij_arr)
print('Processing Blood Cancer Dataset')
dlp.applyDL(dataset_files[2])
acc3 = cf.findAccuracy(dataset_files[2], start_rows[2], end_rows[2], classes_arr[2],1,dij_arr)
acc = max([acc1,acc2,acc3]);
print('Final accuracy of ensemble learning %0.04f %%' % (acc * 100))
