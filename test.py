# from osgeo import ogr

# wkt1 = "POLYGON ((1208064.271243039 624154.6783778917, 1208064.271243039 601260.9785661874, 1231345.9998651114 601260.9785661874, 1231345.9998651114 624154.6783778917, 1208064.271243039 624154.6783778917))"
# wkt2 = "POLYGON ((1199915.6662253144 633079.3410163528, 1199915.6662253144 614453.958118695, 1219317.1067437078 614453.958118695, 1219317.1067437078 633079.3410163528, 1199915.6662253144 633079.3410163528)))"

# poly1 = ogr.CreateGeometryFromWkt(wkt1)
# poly2 = ogr.CreateGeometryFromWkt(wkt2)

# intersection = poly1.Intersection(poly2)

# print(intersection.ExportToWkt())
# import shapely
from shapely.geometry import Polygon
import numpy as np

# [280,100],[280,200],[405,200],[405,100]

#check collision

# pts = np.array([[280,100],[280,200],[405,200],[405,100]], np.int32)
# p1=Polygon(pts)
# p2=Polygon([[280,100],[280,200],[405,200],[405,150]])
# p3=p1.intersection(p2)
# print(p3) # result: POLYGON ((0.5 0.5, 1 1, 1 0, 0.5 0.5))
# print(p3.area) # result: 0.25

# get axis
# import numpy as np
# import matplotlib.pyplot as plt
# import mpldatacursor
# import cv2

# data = cv2.imread("../images/illegal.jpg")

# fig, ax = plt.subplots()
# ax.imshow(data, interpolation='none')

# mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))
# plt.show()

!pip install dlib
import dlib