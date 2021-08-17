import sys
from pytesseract import image_to_string
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# Get a image from search
img_link = 'http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg'
img = requests.get(img_link)
img = Image.open(BytesIO(img.content))

#show image

img_arr = np.array(img)
plt.imshow(img_arr)
plt.show()

'''
About "reverse_geocoder" 
There are 2 main math forms: encode a geographical coordinates from a address
and reverse encoding
Both of that can be solved base on API of google map or OpenStreetMap 
'''