# Code update from: https://github.com/YashGunjal/asm

import numpy as np
import sys
import math
import glob
import os
from utils.read_write_pts import read_pts_file
import cv2
import warnings
warnings.filterwarnings("ignore")

def drange(start, stop, step):
  r = start
  while r < stop:
    yield r
    r += step

class Point ( object ):
  """ Class to represent a point in 2d cartesian space """
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __add__(self, p):
    """ Return a new point which is equal to this point added to p
    :param p: The other point
    """
    return Point(self.x + p.x, self.y + p.y)

  def __truediv__(self, i):
    return Point(self.x/i, self.y/i)

  def __eq__(self, other):
    return self.x == other.x and self.y == other.y

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    """return a string representation of this point. """
    return '(%f, %f)' % (self.x, self.y)

  def dist(self, p):
    """ Return the distance of this point to another point

    :param p: The other point
    """
    return math.sqrt((p.x - self.x)**2 + (p.y - self.y)**2)

class Shape ( object ):
  """ Class to represent a shape.  This is essentially a list of Point
  objects
  """
  def __init__(self, pts = []):
    self.pts = pts
    self.num_pts = len(pts)

  def __add__(self, other):
    """ Operator overloading so that we can add one shape to another
    """
    s = Shape([])
    for i,p in enumerate(self.pts):
      s.add_point(p + other.pts[i])
    return s

  def __truediv__(self, i):
    """ Division by a constant.
    Each point gets divided by i
    """
    s = Shape([])
    for p in self.pts:
      s.add_point(p/i)
    return s

  def __eq__(self, other):
    for i in range(len(self.pts)):
      if self.pts[i] != other.pts[i]:
        return False
    return True

  def __ne__(self, other):
    return not self.__eq__(other)

  def add_point(self, p):
    self.pts.append(p)
    self.num_pts += 1

  def transform(self, t):
    s = Shape([])
    for p in self.pts:
      s.add_point(p + t)
    return s

  """ Helper methods for shape alignment """
  def __get_X(self, w):
    return sum([w[i]*self.pts[i].x for i in range(len(self.pts))])
  def __get_Y(self, w):
    return sum([w[i]*self.pts[i].y for i in range(len(self.pts))])
  def __get_Z(self, w):
    return sum([w[i]*(self.pts[i].x**2+self.pts[i].y**2) for i in range(len(self.pts))])
  def __get_C1(self, w, s):
    return sum([w[i]*(s.pts[i].x*self.pts[i].x + s.pts[i].y*self.pts[i].y) \
        for i in range(len(self.pts))])
  def __get_C2(self, w, s):
    return sum([w[i]*(s.pts[i].y*self.pts[i].x - s.pts[i].x*self.pts[i].y) \
        for i in range(len(self.pts))])

  def get_alignment_params(self, s, w):
    """ Gets the parameters required to align the shape to the given shape
    using the weight matrix w.  This applies a scaling, transformation and
    rotation to each point in the shape to align it as closely as possible
    to the shape.

    This relies on some linear algebra which we use numpy to solve.

    [ X2 -Y2   W   0][ax]   [X1]
    [ Y2  X2   0   W][ay] = [Y1]
    [ Z    0  X2  Y2][tx]   [C1]
    [ 0    Z -Y2  X2][ty]   [C2]

    We want to solve this to find ax, ay, tx, and ty

    :param shape: The shape to align to
    :param w: The weight matrix
    :return x: [ax, ay, tx, ty]
    """

    X1 = s.__get_X(w)
    X2 = self.__get_X(w)
    Y1 = s.__get_Y(w)
    Y2 = self.__get_Y(w)
    Z = self.__get_Z(w)
    W = sum(w)
    C1 = self.__get_C1(w, s)
    C2 = self.__get_C2(w, s)

    a = np.array([[ X2, -Y2,   W,  0],
                  [ Y2,  X2,   0,  W],
                  [  Z,   0,  X2, Y2],
                  [  0,   Z, -Y2, X2]])

    b = np.array([X1, Y1, C1, C2])
    # Solve equations
    # result is [ax, ay, tx, ty]
    return np.linalg.solve(a, b)

  def apply_params_to_shape(self, p):
    new = Shape([])
    # For each point in current shape
    for pt in self.pts:
      new_x = (p[0]*pt.x - p[1]*pt.y) + p[2]
      new_y = (p[1]*pt.x + p[0]*pt.y) + p[3]
      new.add_point(Point(new_x, new_y))
    return new

  def align_to_shape(self, s, w):
    p = self.get_alignment_params(s, w)
    return self.apply_params_to_shape(p)

  def get_vector(self):
    vec = np.zeros((self.num_pts, 2))
    for i in range(len(self.pts)):
      vec[i,:] = [self.pts[i].x, self.pts[i].y]   #vec[i,:]：  vec的第i个tuple
    return vec.flatten()   #拉平

  def get_normal_to_point(self, p_num):
    # Normal to first point
    x = 0; y = 0; mag = 0
    if p_num == 0:
      x = self.pts[1].x - self.pts[0].x
      y = self.pts[1].y - self.pts[0].y
    # Normal to last point
    elif p_num == len(self.pts)-1:
      x = self.pts[-1].x - self.pts[-2].x
      y = self.pts[-1].y - self.pts[-2].y
    # Must have two adjacent points, so...
    else:
      x = self.pts[p_num+1].x - self.pts[p_num-1].x
      y = self.pts[p_num+1].y - self.pts[p_num-1].y
    mag = math.sqrt(x**2 + y**2)
    return (-y/mag, x/mag)

  @staticmethod
  def from_vector(vec):
    s = Shape([])
    for i,j in np.reshape(vec, (-1,2)):
      s.add_point(Point(i, j))
    return s

class PointsReader ( object ):
  """ Class to read from files provided on Tim Cootes's website."""
  @staticmethod
  def read_points_file(filename):
    """ Read a .pts file, and returns a Shape object """
    s = Shape([])
    points,num_pts,width,height = read_pts_file(filename)
    for pt in points:
        s.add_point(Point(float(pt[0]/width), float(pt[1]/height))) 
    if s.num_pts != num_pts:
      print("Unexpected number of points in file.  "+\
      "Expecting %d, got %d" % (num_pts, s.num_pts))
    return s

  @staticmethod
  def read_directory(dirname):
    """ Reads an entire directory of .pts files and returns
    them as a list of shapes
    """
    pts = []
    for file in glob.glob(os.path.join(dirname, "*.pts")):
      pts.append(PointsReader.read_points_file(file))
    return pts

class PointsReader_cluster ( object ):##128 point
  """ Class to read from files provided on Tim Cootes's website."""
  @staticmethod
  def read_point(point,number_point): #point ,258
    if number_point == 128:
      p = point[:256].reshape(-1,2)
      width = point[256]
      height = point[257]
      s = Shape([])
      for pt in p:
          s.add_point(Point(float(pt[0]/width), float(pt[1]/height))) 
      if s.num_pts != 128:
        print("Unexpected number of points in file.  "+\
        "Expecting %d, got %d" % (128, s.num_pts))
    elif number_point == 64:
      p = point[:128].reshape(-1,2)
      width = point[128]
      height = point[129]
      s = Shape([])
      for pt in p:
          s.add_point(Point(float(pt[0]/width), float(pt[1]/height))) 
      if s.num_pts != 64:
        print("Unexpected number of points in file.  "+\
        "Expecting %d, got %d" % (64, s.num_pts))
    return s

  @staticmethod
  def read_points(points,number_point):
    """ Reads an entire directory of .pts files and returns
    them as a list of shapes
    """
    pts = []
    for p in points:
      pts.append(PointsReader_cluster.read_point(p,number_point))
    return pts

class ModelFitter:
  """
  Class to fit a model to an image

  :param asm: A trained active shape model
  :param image: An OpenCV image
  :param t: A transformation to move the shape to a new origin
  """
  def __init__(self, asm, image, t=Point(100.0,100.0)):
    self.image = image
    self.g_image = []
    for i in range(0,4):
      self.g_image.append(self.__produce_gradient_image(image, 2**i))
    self.asm = asm
    # Copy mean shape as starting shape and transform it to origin
    self.shape = Shape.from_vector(asm.mean).transform(t)
    # And resize shape to fit image if required
    if self.__shape_outside_image(self.shape, self.image):
      self.shape = self.__resize_shape_to_fit_image(self.shape, self.image)

  def __shape_outside_image(self, s, i):
    for p in s.pts:
      if p.x >= i.shape[1] or p.x < 0 or p.y >= i.shape[0] or p.y < 0:
        return True
    return False

  def __resize_shape_to_fit_image(self, s, i):
    # Get rectagonal boundary orf shape
    min_x = min([pt.x for pt in s.pts])
    min_y = min([pt.y for pt in s.pts])
    max_x = max([pt.x for pt in s.pts])
    max_y = max([pt.y for pt in s.pts])

    # If it is outside the image then we'll translate it back again
    if min_x > i.shape[1]: min_x = 0
    if min_y > i.shape[0]: min_y = 0
    ratio_x = (i.shape[1]-min_x) / (max_x - min_x)
    ratio_y = (i.shape[0]-min_y) / (max_y - min_y)
    new = Shape([])
    for pt in s.pts:
      new.add_point(Point(pt.x*ratio_x if ratio_x < 1 else pt.x, \
                          pt.y*ratio_y if ratio_y < 1 else pt.y))
    return new

  def __produce_gradient_image(self, i, scale):
    size = i.shape[:2]
    grey_image = np.zeros((size[1], size[0]), dtype=np.uint8)
    size = [s/scale for s in size]
    
    grey_image_small = np.zeros((int(size[1]), int(size[0])), dtype=np.uint8)
    
    grey_image = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) # 使用 cv2.cvtColor 函数将彩色图像转换为灰度图像
    
    dy_dx = np.zeros((i.shape[0], i.shape[1]), dtype=np.int16)
    
    dy_dx = cv2.Sobel(grey_image, cv2.CV_16S, 1, 1, ksize=3)# 计算图像的水平和垂直梯度

    grey_image = cv2.convertScaleAbs(dy_dx)
    grey_image_small = cv2.resize(grey_image, (int(size[1]), int(size[0])), interpolation=cv2.INTER_NEAREST)   #将grey_image下采样当(i=3时下采样率分别为1,2,4,8)到small
    grey_image = cv2.resize(grey_image_small, (int(size[1]), int(size[1])), interpolation=cv2.INTER_NEAREST)   #再将small 按当前下采样尺度赋回grey_image
    return grey_image

  def do_iteration(self, scale):
    """ Does a single iteration of the shape fitting algorithm.
    This is useful when we want to show the algorithm converging on
    an image

    :return shape: The shape in its current orientation
    """

    # Build new shape from max points along normal to current
    # shape
    s = Shape([])
    for i, pt in enumerate(self.shape.pts):
      s.add_point(self.__get_max_along_normal(i, scale))

    new_s = s.align_to_shape(Shape.from_vector(self.asm.mean), self.asm.w)

    var = new_s.get_vector() - self.asm.mean
    new = self.asm.mean
    for i in range(len(self.asm.evecs.T)):
      b = np.dot(self.asm.evecs[:,i],var)
      max_b = 2*math.sqrt(self.asm.evals[i])
      b = max(min(b, max_b), -max_b)

      new = new + self.asm.evecs[:,i]*b

    self.shape = Shape.from_vector(new).align_to_shape(s, self.asm.w)

  def __get_max_along_normal(self, p_num, scale):
    """
    在这段代码中,scale 是用于确定当前梯度图像的尺度。梯度图像的尺度决定了形状匹配算法搜索边缘的精细程度。较大的 scale 值表示使用较低分辨率的梯度图像进行搜索，而较小的 scale 值表示使用较高分辨率的梯度图像进行搜索。通过调整 scale 值，可以控制搜索的精度和计算的效率。
    在该函数中,scale 的值用于确定在搜索最大边缘响应时的搜索范围。较大的 scale 值将导致搜索范围更广，从而更容易找到具有最大边缘响应的点。较小的 scale 值将导致搜索范围更窄，从而更精确地定位边缘。
    
    
    通过在不同的尺度上进行搜索，可以在不同的精细程度上调整形状的匹配结果。通常，在初始阶段使用较大的 scale 值进行快速搜索，然后在后续的迭代中逐渐减小 scale 值进行精细调整。
    """
    """ Gets the max edge response along the normal to a point

    :param p_num: Is the number of the point in the shape
    """

    norm = self.shape.get_normal_to_point(p_num)      
    p = self.shape.pts[p_num]

    # Find extremes of normal within the image 在图像中找到法线的极值
    # Test x first
    min_t = -p.x / norm[0]
    if p.y + min_t*norm[1] < 0:
      min_t = -p.y / norm[1]
    elif p.y + min_t*norm[1] > self.image.shape[0]:
      min_t = (self.image.shape[0] - p.y) / norm[1]

    # X first again
    max_t = (self.image.shape[1] - p.x) / norm[0]
    if p.y + max_t*norm[1] < 0:
      max_t = -p.y / norm[1]
    elif p.y + max_t*norm[1] > self.image.shape[0]:
      max_t = (self.image.shape[0] - p.y) / norm[1]

    # Swap round if max is actually larger...
    tmp = max_t
    max_t = max(min_t, max_t)
    min_t = min(min_t, tmp)

    # Get length of the normal within the image
    x1 = min(p.x+max_t*norm[0], p.x+min_t*norm[0])
    x2 = max(p.x+max_t*norm[0], p.x+min_t*norm[0])
    y1 = min(p.y+max_t*norm[1], p.y+min_t*norm[1])
    y2 = max(p.y+max_t*norm[1], p.y+min_t*norm[1])
    l = math.sqrt((x2-x1)**2 + (y2-y1)**2)

    img = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
    img = self.g_image[scale].copy()
    #cv.Circle(img, \
    #    (int(norm[0]*min_t + p.x), int(norm[1]*min_t + p.y)), \
    #    5, (0, 0, 0))
    #cv.Circle(img, \
    #    (int(norm[0]*max_t + p.x), int(norm[1]*max_t + p.y)), \
    #    5, (0, 0, 0))

    # Scan over the whole line
    max_pt = p
    max_edge = 0

    # Now check over the vector
    #v = min(max_t, -min_t)
    #for t in drange(min_t, max_t, (max_t-min_t)/l):
    search = 20+scale*10
    # Look 6 pixels to each side too
    for side in range(-6, 6):
      # Normal to normal...
      new_p = Point(p.x + side*-norm[1], p.y + side*norm[0])
      for t in drange(-search if -search > min_t else min_t, \
                       search if search < max_t else max_t , 1):

        x = int((norm[0]*t + new_p.x)/2**scale)
        y = int((norm[1]*t + new_p.y)/2**scale)
        if x < 0 or x > self.image.shape[0] or y < 0 or y > self.image.shape[1]:
          continue
#        cv.Circle(img, (x, y), 3, (100,100,100))
        #print x, y, self.g_image.width, self.g_image.height
        if self.g_image[scale][y-1, x-1] > max_edge:
          max_edge = self.g_image[scale][y-1, x-1]
          max_pt = Point(new_p.x + t*norm[0], new_p.y + t*norm[1])

#    for point in self.shape.pts:
#      cv.Circle(img, (int(point.x), int(point.y)), 3, (255,255,255))
##
#    cv.Circle(img, (int(max_pt.x), int(max_pt.y)), 3, (255,255,255))
##
#    cv.NamedWindow("Scale", cv.CV_WINDOW_AUTOSIZE)
#    cv.ShowImage("Scale",img)
#    cv.WaitKey()
#
    return max_pt

class ActiveShapeModel:
  """
  """
  def __init__(self, shapes = [],t_max = 0,padding = False):
    self.shapes = shapes
    self.t_max = t_max
    self.padding = padding
    # Make sure the shape list is valid
    self.__check_shapes(shapes)
    # Create weight matrix for points
    print ("Calculating weight matrix...")
    self.w = self.__create_weight_matrix(shapes)
    # Align all shapes
    print ("Aligning shapes with Procrustes analysis...")
    self.shapes = self.__procrustes(shapes)
    print ("Constructing model...")
    # Initialise this in constructor
    (self.evals, self.evecs, self.mean, self.modes) = \
        self.__construct_model(self.shapes,self.t_max,self.padding)

  def __check_shapes(self, shapes):
    """ Method to check that all shapes have the correct number of
    points """
    if shapes:
      num_pts = shapes[0].num_pts
      for shape in shapes:
        if shape.num_pts != num_pts:
          raise Exception("Shape has incorrect number of points")

  def __get_mean_shape(self, shapes):
    s = shapes[0]
    for shape in shapes[1:]:
      s = s + shape
    ms = s / len(shapes)
    return ms

  def __construct_model(self, shapes,t_max = 0,padding = False):
    """ Constructs the shape model
    """
    shape_vectors = np.array([s.get_vector() for s in self.shapes])  
    mean = np.mean(shape_vectors, axis=0)

    # Move mean to the origin
    # FIXME Clean this up...
    mean = np.reshape(mean, (-1,2)) 

    

    # Produce covariance matrix
    if len(shape_vectors) == 1:
        t = 1
        if padding == True:
          t = t_max 
        evals = np.ones(t)
        evecs = np.tile(shape_vectors, (t, 1)).transpose(1,0)
        return (evals[:t], evecs[:,:t], mean, t)
    cov = np.cov(shape_vectors, rowvar=0)
    # Find eigenvalues/vectors of the covariance matrix
    evals, evecs = np.linalg.eig(cov)

    # Find number of modes required to describe the shape accurately
    t = 0
    #print(len(evals))
    for i in range(len(evals)):
      if sum(evals[:i]) / sum(evals) < 0.995:   
        t = t + 1
      else: break
    
    print ("Constructed model with %d modes of variation" % t)
    if padding == True:
      t = t_max 
    return (evals[:t], evecs[:,:t], mean, t)

  def generate_example(self, b):
    """ b is a vector of floats to apply to each mode of variation
    """
    # Need to make an array same length as mean to apply to eigen
    # vectors
    full_b = np.zeros(len(self.mean))
    for i in range(self.modes): full_b[i] = b[i]

    p = self.mean
    for i in range(self.modes): p = p + full_b[i]*self.evecs[:,i]

    # Construct a shape object
    return Shape.from_vector(p)

  def __procrustes(self, shapes):
    """ This function aligns all shapes passed as a parameter by using
    Procrustes analysis

    :param shapes: A list of Shape objects
    """
    # First rotate/scale/translate each shape to match first in set

    shapes[1:] = [s.align_to_shape(shapes[0], self.w) for s in shapes[1:]]        

    # Keep hold of a shape to align to each iteration to allow convergence
    a = shapes[0]
    if len(shapes) == 1:
        return shapes
    trans = np.zeros((4, len(shapes)))
    converged = False
    current_accuracy = sys.maxsize
    while not converged:
      # Now get mean shape
      mean = self.__get_mean_shape(shapes)
      # Align to shape to stop it diverging
      mean = mean.align_to_shape(a, self.w)
      # Now align all shapes to the mean
      for i in range(len(shapes)):
        # Get transformation required for each shape
        trans[:, i] = shapes[i].get_alignment_params(mean, self.w)
        # Apply the transformation
        shapes[i] = shapes[i].apply_params_to_shape(trans[:,i])

      # Test if the average transformation required is very close to the
      # identity transformation and stop iteration if it is
      accuracy = np.mean(np.array([1, 0, 0, 0]) - np.mean(trans, axis=1))**2
      # If the accuracy starts to decrease then we have reached limit of precision
      # possible
      if accuracy > current_accuracy: converged = True
      else: current_accuracy = accuracy
    return shapes

  def __create_weight_matrix(self, shapes):  
    """ Private method to produce the weight matrix which corresponds
    to the training shapes

    :param shapes: A list of Shape objects
    :return w: The matrix of weights produced from the shapes
    """
    # Return empty matrix if no shapes
    if not shapes:
      return np.array()
    # First get number of points of each shape
    num_pts = shapes[0].num_pts
    # We need to find the distance of each point to each
    # other point in each shape.
    distances = np.zeros((len(shapes), num_pts, num_pts))
    for s, shape in enumerate(shapes):
      for k in range(num_pts):
        for l in range(num_pts):
          distances[s, k, l] = shape.pts[k].dist(shape.pts[l])

    # Create empty weight matrix
    w = np.zeros(num_pts)
    # calculate range for each point
    for k in range(num_pts):
      for l in range(num_pts):
        # Get the variance in distance of that point to other points
        # for all shapes
        w[k] += np.var(distances[:, k, l])
    # Invert weights
    return 1/w

