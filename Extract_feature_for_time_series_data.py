'''
One-hot_encoding is a important method to encode time series parameter.
A example for this is that determine every day in a week
using weekday()
'''

# from datetime import datetime
#
# def parser(x):
#     #To know format strftime of a characterstring we need to find in string format time table
#     return  datetime.strptime(x,'%Y-%m-%d %H:%M:%S' )
# dataset['created'] = dataset['created'].map(lambda x: parser(x))
# #Check time format
#
# for i,k in zip(dataset.columns, dataset.dtypes):
#     print(f'{i}: {k}')