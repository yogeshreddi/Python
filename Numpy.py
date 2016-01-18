
# coding: utf-8

# # Numpy
# ### Creating Arrays

# In[1]:

import numpy as np


# In[2]:

my_list = [1,2,3,4]


# In[3]:

my_array = np.array(my_list)


# In[4]:

#Array from a list
print(my_array)
my_array


# In[5]:

#Multidiemensional Array 
l1 = [1,2,3,4]
l2 = [5,6,7,8]
my_array1 = np.array(l1,l2) #Will give an error here 


# In[6]:

# Above error corrected here two lists can be formed an array if and only if both of them are elements of another tuple of list 
my_array1 = np.array((l1,l2))
my_array1


# In[7]:

my_array1.shape


# In[8]:

my_array1.dtype


# In[9]:

my_array2 = np.zeros(5)


# In[10]:

my_array2


# In[11]:

my_array2.dtype


# In[12]:

my_array3 = np.zeros(5,5) # Will give an error here


# In[13]:

my_array3 = np.zeros((5,5))


# In[14]:

my_array3


# In[15]:

my_arrya4 = np.ones((2,6))


# In[16]:

my_arrya4


# In[17]:

my_array5 = np.empty([2,3])


# In[18]:

print(my_array5)
my_array5.dtype


# In[19]:

my_array5 = np.eye(5)


# In[20]:

my_array5


# In[21]:

np.arange(1,10)


# In[22]:

np.arange(10)


# In[23]:

np.arange(2,101,2)


# # Using arrays as scalars

# In[24]:

arr1 = np.array([[1,2,3,4],[8,9,10,11]])


# In[25]:

arr1


# In[26]:

arr1*arr1


# In[27]:

arr1+arr1


# In[28]:

arr1-arr1


# In[29]:

arr1/arr1


# In[30]:

1/arr1


# In[31]:

arr1**4


# # Indexing Arrays

# In[32]:

arr2 = np.arange(0,11)


# In[33]:

arr2


# In[34]:

arr2[0]


# In[35]:

arr2[1]


# In[36]:

arr2[1:8]


# In[37]:

slice_of_arr2 = arr2[0:5]


# In[38]:

slice_of_arr2=77
slice_of_arr2


# In[39]:

slice_of_arr3 = arr2[0:5]


# In[40]:

slice_of_arr3[:] = 77
slice_of_arr3


# In[41]:

#Still effects our original array
arr2


# In[42]:

#to make sure we are making a copy of original not just assigning the same memory location
arr2_copy = arr2[5:8].copy()


# In[43]:

arr2_copy[:] = 66
arr2_copy 


# In[44]:

#Doesn't effect the original array as we made a copy, instead of just assigning the array
arr2


# In[45]:

arr2d = np.array(([5,10,15,20],[25,30,35,40],[45,50,55,60]))
arr2d


# In[46]:

#calling by index array_name[row][column]
print(arr2d[0])
print(arr2d[2][3])


# In[47]:

slice_of_arr2d = arr2d[0:2,2:4].copy()


# In[48]:

slice_of_arr2d


# In[49]:

#Fancy indexing

arr2_d = np.zeros((10,10))


# In[50]:

arr2_d


# In[51]:

arr2_d.shape[0]


# In[52]:

for i in range(arr2_d.shape[0]):
    arr2_d[i] = i


# In[53]:

arr2_d


# In[54]:

for i in range(arr2_d.shape[0]):
    for j in range(arr2_d.shape[1]):
        arr2_d[i][j] = i*j


# In[55]:

arr2_d


# In[56]:

arr2_d[2,8]


# In[57]:

arr2_d[(2,8)]


# In[58]:

arr2_d[[2,8]] #returns the second row and 8th row


# In[59]:

arr2_d


# In[60]:

np.array(['1',1,'2',2]).dtype


# In[61]:

np.arange(0,11,10)


# In[62]:

np.zeros([10,10])


# # Arrays Transposition/Transposing

# In[63]:

arr = np.arange(50).reshape([10,5])


# In[64]:

arr 


# In[65]:

#Transpose of arr
arr.T


# In[66]:

#Caluculating dot product
np.dot(arr.T,arr)


# In[67]:

arr3d = np.arange(12).reshape((3,2,2))


# In[68]:

arr3d


# In[69]:

arr3d.transpose((1,0,2))


# In[70]:

np.array([[1,2,3]]).swapaxes(1,0)


# # Universal Array Functions
# ## Functions which can be applied to every element in an array.
# ###

# In[71]:

arr_univ = np.arange(37)
arr_univ


# In[72]:

#Returns the square root of every element in the array
np.sqrt(arr_univ)


# In[73]:

np.exp(arr_univ)


# In[74]:

#Warning has ocured as we tried to evaluate the log (0)..
np.log(arr_univ)


# In[75]:

# to generate an array of normal disturbeted numbers randomly
A= np.random.randn(10)
A


# In[76]:

B = np.random.randn(10)
B


# ### Binary Funcions

# In[77]:

np.add(A,B)


# In[78]:

A+B


# In[79]:

#Two arrays passed should have the same shape, and array with the maximum value at each possition will be return.
np.maximum(A,B)


# In[80]:

C= np.random.randn(10)
C


# In[81]:

#Two arrays passed should have the same shape, and array with the minumum value at each possition will be return.
np.minimum(A,B)


# In[82]:

D = np.empty(10)
D


# In[83]:

#Maximum matrix is evaluated from A,B and stored in D
np.maximum(A,B,D)


# In[84]:

D


# In[85]:

import webbrowser as wb
url = 'http://docs.scipy.org/doc/numpy/reference/'
wb.open(url)


# ## Array Processing

# In[86]:

# Provides a MATLAB-like plotting framework.
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# to see the visualisation


# In[87]:

points = np.arange(-5, 5, 0.01)
points


# In[88]:

dx,dy = np.meshgrid(points,points)


# In[89]:

dx


# In[90]:

dy


# In[91]:

z = np.sin(dx)+np.sin(dy)
z


# In[92]:

plt.imshow(dx)


# In[93]:

plt.imshow(dy)


# In[94]:

plt.imshow(z)


# In[95]:

plt.imshow(z)
plt.colorbar()
plt.title('sin(x)+sin(y)')


# In[96]:

A = np.array([1,2,3,4])
B = np.array([100,200,300,400])
conditions =np.array([True,True,False,True])


# In[97]:

# Forming an array based on condtion to A,B.. If condition is True element is taken from A otherwise From B
answer =  [(A_val if cond else B_val) for A_val,B_val,cond in zip(A,B,conditions)]


# In[98]:

answer


# In[99]:

# Above code  written in a single inbuilt function call
answer2 = np.where(conditions,A,B)


# In[100]:

answer2


# In[101]:

arr4 = np.random.randn(11)


# In[102]:

arr4


# In[103]:

np.where(arr4>0,arr4,0)


# In[104]:

X = np.random.randn(10)
Y = np.random.randn(10)


# In[105]:

print(X)
print(Y)


# In[106]:

# Array fromed by selecting the highest value among X,Y for each position 
np.where(X>Y,X,Y)


# In[107]:

# Array fromed by selecting the highest value among X,Y for each position and also replace the highest values whihc are negatuves with zero(0)
np.where(np.where(X>Y,X,Y)<0,0,np.where(X>Y,X,Y))


# In[108]:

# Above line of code written using maximum method 
np.where(np.maximum(X,Y)<0,0,np.maximum(X,Y))


# In[109]:

arr5 = np.array(([1,2,3],[4,5,6],[7,8,9]))
arr5


# In[110]:

#to find the sun of elements in the same column(i.e sum of elements in each row in the same column)
arr5.sum(0)


# In[111]:

#to find the sun of elements in the same rows(i.e sum of elements in each column in the same row)
arr5.sum(1)


# In[112]:

#to find mean,standard deviation,and variance

# the below will give mean of elements in the same colun
arr5.mean(0) 


# In[113]:

# Mean of complete array
arr5.mean() 


# In[114]:

#Standard deviation of comeplete array
arr5.std()


# In[115]:

# similarly variance
arr5.var()


# In[116]:

np.var(arr5)


# #### Boolean 

# In[117]:

arr_bool = np.array([True,False,False,True,False])


# In[118]:

np.any(arr_bool)


# In[119]:

np.all(arr_bool)


# In[120]:

arr_bool.any()


# ### Sorting

# In[121]:

arr_sort = np.random.randn(10)
arr_sort


# In[122]:

arr_sort.sort()
arr_sort


# ### unique or not unique

# In[123]:

numbers = np.array([1,22,1,45,22,98,9,7,1,9])
numbers


# In[124]:

# returns all the values avilable in the array
np.unique(numbers)


# In[125]:

#check if the numbers given int the array are presnt in numbers array or not.. Returns True or false for all the elements in the array passed
np.in1d([1,22,45,76],numbers)


# # Array Input and Output

# In[126]:

#to save an array and to load it after on

arr6 = np.array([0,1,2,3,4])
np.save('myarray',arr6)
arr6


# In[127]:

# changing the elemtns in the array arr6
arr6 = np.arange(10)
arr6 


# In[128]:

#lets now load the 'myarray' file into a new array

arr7 = np.load('myarray.npy')
arr7


# In[129]:

#lets now save bth arr6 and arr7 to a zip file together

np.savez('myarrayzip',x=arr6,y=arr7)


# In[130]:

#loading the saved multiple arrays
saved_arrays = np.load('myarrayzip.npz')


# In[131]:

#getting the inndividula array loaded arrays
saved_arrays['x']


# In[132]:

saved_arrays['y']


# In[133]:

# saving arrays to .txt file

arr8 = np.array(([1,2,3],[4,5,6],[7,8,9]))
arr8


# In[134]:

# saving arrays to .txt file with comma as seperater/delimiter
np.savetxt('mytextarray.txt',arr8,delimiter=',')


# In[135]:

#Loading the array from .txt file
text_array = np.loadtxt('mytextarray.txt',delimiter=',')


# In[136]:

text_array


# In[137]:

# For more information execute the below command, it will redirect you to SciPy.org page with A to Z everything about NumPy
import webbrowser as wb
url = 'http://docs.scipy.org/doc/numpy/reference/'
wb.open(url)


# In[ ]:



