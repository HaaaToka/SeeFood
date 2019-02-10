import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import copy

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from matplotlib import cm
from PIL import Image
import pandas as pd
from collections import defaultdict
import csv

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = '/Users/okanalan/Desktop/ML/bbm406-project-seefood/Dataset/Inference Graphs/frcnn_inception'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('legacy/data', 'object-detection.pbtxt')


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



"""dicSide = { 'apple':[], 'banana':[], 'bread':[], 'bun':[],
        'doughnut':[], 'egg':[], 'fired_dough_twist':[],
        'grape':[], 'lemon':[], 'litchi':[], 'mango':[],
        'mooncake':[], 'orange':[], 'peach':[], 'pear':[],
        'plum':[], 'qiwi':[], 'sachima':[], 'tomato':[],'mix':[] }
dicTop = { 'apple':[], 'banana':[], 'bread':[], 'bun':[],
        'doughnut':[], 'egg':[], 'fired_dough_twist':[],
        'grape':[], 'lemon':[], 'litchi':[], 'mango':[],
        'mooncake':[], 'orange':[], 'peach':[], 'pear':[],
        'plum':[], 'qiwi':[], 'sachima':[], 'tomato':[],'mix':[] }"""
dicSd={}
dicTp={}

PATH_TO_TEST_IMAGES_DIR = 'legacy/images/test'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, file) for file in os.listdir(PATH_TO_TEST_IMAGES_DIR) if 'JPG' in file]
#TEST_IMAGE_PATHS=[os.path.join(PATH_TO_TEST_IMAGES_DIR,"fired_dough_twist001T(7).JPG")]
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


for file in os.listdir(PATH_TO_TEST_IMAGES_DIR):
    if 'JPG' in file:
        sito = file.split('(')[0][-1]
        cate = file.split('0')[0]
        speci = file.split('(')[0][:-1]
        if cate not in dicSd.keys():
            dicSd[cate]=defaultdict(list)
            dicTp[cate]=defaultdict(list)
        """if sito.upper()=='S':
            dicSd[cate][speci].append(file)
        else:
            dicTp[cate][speci].append(file)"""
print(dicSd)

"""for ke in dicSd.keys():
    print(ke,'-> ',len(dicSd[ke]),len(dicTp[ke]))
    for ke2,val in dicSd[ke].items():
        print('\t',ke2,'-> ',len(val),len(dicTp[ke][ke2]),"---",len(val)-len(dicTp[ke][ke2]))
print(dicTp['grape'],"\n",dicSd['grape'])"""


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

shape = {'ellipsoid':['apple','egg','lemon','orange','peach','plum','qiwi','tomato','mix'],
        'column':['bread','grape','mooncake','sachima'],
        'irregular':['banana','bun','doughnut','fired_dough_twist','litchi','mango','pear']}

def findShape(shapes,exp):
    for k,v in shapes.items():
        for elem in v:
            if exp==elem:
                return k

xls_to_dict={}
xls = pd.ExcelFile("density.xls")
for i in range(20):
    sheetX = xls.parse(i) #2 is the sheet number
    idd = sheetX['id']
    typee = sheetX['type']
    vol = sheetX['volume(mm^3)']
    wei = sheetX['weight(g)']
    if 'mix' in idd[0]:
        xls_to_dict['mix']={}
    else:
        xls_to_dict[typee[0]]={}
    for k in range(len(idd)):
        if 'mix' in idd[0]:
            xls_to_dict['mix'][idd[k]]=[vol[k],wei[k]]
        else:
            xls_to_dict[typee[0]][idd[k]]=[vol[k],wei[k]]


dicSide=defaultdict(list)
dicTop=defaultdict(list)

print(len(TEST_IMAGE_PATHS),"<<<<TEST_IMAGE_PATHS LENNNN")

breaker=0
cnt=0
whichh='mix'
for image_path in TEST_IMAGE_PATHS:
  cnt+=1
  print(cnt,"<><><numero><><>")
  print(image_path)
  if whichh in image_path:
      continue
  image = Image.open(image_path)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
  output_dict = run_inference_for_single_image(image_np, detection_graph)
  # Visualization of the results of a detection.
  #print(len(output_dict['detection_boxes']),"<<<<<<<<<",image_path.split('/')[-1])
  coordinates = vis_util.return_coordinates(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        output_dict['detection_scores'],
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8,
                        min_score_thresh=0.5)
  #[ymin, ymax, xmin, xmax, (box_to_score_map[box]*100)] y->height x->width
  #print(coordinates,"<<<<<<<<<",image_path.split('/')[-1])
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)
  #plt.figure(figsize=IMAGE_SIZE)
  #plt.imshow(image_np)
  #plt.show()

  #[xmin,ymin,xmax,ymax]

  for obbj,vl in coordinates.items():

      area = (coordinates[obbj][2],
              coordinates[obbj][0],
              coordinates[obbj][3],
              coordinates[obbj][1])
      #print(area)

      imp = Image.open(image_path)
      cropped_img = imp.crop(area)
      cropped_img.save('temp.png')

      clone_img = cv2.imread('temp.png')
      #print(clone_img.shape)
      mask = np.zeros(clone_img.shape[:2],np.uint8)
      bgdModel = np.zeros((1,65),np.float64)
      fgdModel = np.zeros((1,65),np.float64)
      rect=(1,1,clone_img.shape[1]-1,clone_img.shape[0]-2)
      cv2.grabCut(clone_img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
      mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
      clone_img = clone_img*mask2[:,:,np.newaxis]
      im = Image.fromarray((clone_img * 255).astype(np.uint8))
      im.save('temp2.png')
      #plt.imshow(clone_img),plt.colorbar()
      #plt.show()

      pixelImage = cv2.imread('temp2.png',0)
      colored_pixel = cv2.countNonZero(pixelImage)
      #print(colored_pixel,pixelImage.shape[0]*pixelImage.shape[1])

      image_name=image_path.split('/')[-1]
      splitName=image_name.split('(')[0]
      dicName = splitName[:-1]
      dicCategory = dicName.split('0')[0]
      #print(xls_to_dict[dicCategory][dicName])

      #print(image_name,"\n",splitName,"\n",dicName,"\n",dicCategory,"\n")
      print("image name",image_name,"\n",
            "objj",obbj,"\n",
            "BOX heigth-width-> ",coordinates[obbj][1]-coordinates[obbj][0],"-",coordinates[obbj][3]-coordinates[obbj][2],"\n",
            "shape-> ",findShape(shape,dicCategory),"\n",
            "foreground pixels-> ",colored_pixel,"\n",
            "real vol-wei",xls_to_dict[dicCategory][dicName])
      """print("image name",image_name,"dicName",dicName)
      print("BOX heigth-width-> ",coordinates[obbj][1]-coordinates[obbj][0],"-",coordinates[obbj][3]-coordinates[obbj][2])
      print("shape-> ",findShape(shape,dicCategory))
      print("foreground pixels-> ",colored_pixel)
      print("real vol-wei",xls_to_dict[dicCategory][dicName],"\n\n")"""

      if splitName[-1].upper()=='S':
          print('>>>>>>>>><dicSide a ',dicName,"\n")
          dicSd[dicCategory][dicName].append([obbj,coordinates[obbj][1]-coordinates[obbj][0],coordinates[obbj][3]-coordinates[obbj][2],findShape(shape,dicCategory),colored_pixel,xls_to_dict[dicCategory][dicName]])
          #dicSide[image_name].append([obbj,coordinates[obbj][1]-coordinates[obbj][0],coordinates[obbj][3]-coordinates[obbj][2],findShape(shape,dicCategory),colored_pixel,xls_to_dict[dicCategory][dicName]])
      else:
          print('>>>>>>>>>dicTop a ',dicName,"\n")
          dicTp[dicCategory][dicName].append([obbj,coordinates[obbj][1]-coordinates[obbj][0],coordinates[obbj][3]-coordinates[obbj][2],findShape(shape,dicCategory),colored_pixel,xls_to_dict[dicCategory][dicName]])
          #dicTop[image_name].append([obbj,coordinates[obbj][1]-coordinates[obbj][0],coordinates[obbj][3]-coordinates[obbj][2],findShape(shape,dicCategory),colored_pixel,xls_to_dict[dicCategory][dicName]])

  """breaker+=1
  if breaker==4:
      break"""





#with open(whichh+'.csv',"w",newline='') as csvfile:
with open('okan.csv',"w",newline='') as csvfile:
    fieldnames=['image_name',
                'Side_coinWidth','Side_coinHeigth',
                'Top_coinWidth','Top_coinHeigth',
                'Side_foodWidth','Side_foodHeigth',
                'Top_foodWidth','Top_foodHeigth',
                'food_label',
                'Side_coinForeground_pixel','Top_coinForeground_pixel',
                'Side_foodForeground_pixel','Top_foodForeground_pixel',
                'realVolume','realDensity']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for k1 in dicSd.keys():
        #print(k1,dicSd[k1],"\n",dicTp[k1],"--------K1\n\n")
        for k2 in dicSd[k1].keys():
            print(k2,dicSd[k1][k2],"\n",dicTp[k1][k2],"--------K2\n")
            l1=len(dicSd[k1][k2])
            l2=len(dicTp[k1][k2])
            mnl = min(l1,l2)
            if mnl%2==1:
                mnl-=1
            for i in range(0,mnl,2):
                print(k1,k2)
                if dicSd[k1][k2][i][0]=='coin':
                    sdCoin = dicSd[k1][k2][i]
                    sdFood=dicSd[k1][k2][i+1]
                else:
                    sdCoin = dicSd[k1][k2][i+1]
                    sdFood=dicSd[k1][k2][i]
                if dicTp[k1][k2][i][0]=='coin':
                    tpCoin=dicTp[k1][k2][i]
                    tpFood=dicTp[k1][k2][i+1]
                else:
                    tpCoin=dicTp[k1][k2][i+1]
                    tpFood=dicTp[k1][k2][i]
                print(sdCoin,sdFood)
                print(tpCoin,tpFood)
                writer.writerow({
                            'image_name':k2,
                            'Side_coinWidth':sdCoin[2],'Side_coinHeigth':sdCoin[1],
                            'Top_coinWidth':tpCoin[2],'Top_coinHeigth':tpCoin[1],
                            'Side_foodWidth':sdFood[2],'Side_foodHeigth':sdFood[1],
                            'Top_foodWidth':tpFood[2],'Top_foodHeigth':tpFood[1],
                            'food_label':tpFood[3],
                            'Side_coinForeground_pixel':sdCoin[4],'Top_coinForeground_pixel':tpCoin[4],
                            'Side_foodForeground_pixel':sdFood[4],'Top_foodForeground_pixel':tpFood[4],
                            'realVolume':sdFood[5][0],'realDensity':sdFood[5][1]/sdFood[5][0]
                })
