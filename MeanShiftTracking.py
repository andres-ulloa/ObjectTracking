
import numpy as np
import cv2 
import math



def computeBackProjection(roi, target):
    roihist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    channels = [0,1,2]
    binsBGR = [8,8,8]
    ranges = [0,256,0,256,0,256]
    bgrhist = cv2.calcHist([roi],channels, None, binsBGR, ranges )
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([target],[0,1],roihist,[0,180,0,256],1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
    ret,thresh = cv2.threshold(dst,50,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(target,thresh)
    #res = np.vstack((target,thresh,res))
    return res


def calculateImageCentroid(img):
    #calculate image moments
    imgM00 = calculateImageMoment(img, 0, 0)
    imgM10 = calculateImageMoment(img, 1, 0)
    imgM01 = calculateImageMoment(img, 0, 1)
    #calculate image centroid coordinates
    imgCentoidCoordinatesX = imgM10/imgM00
    imgCentoidCoordinatesY = imgM01/imgM00
    return (imgCentoidCoordinatesX, imgCentoidCoordinatesY)


def computeOrientation(img,x,y):
    imgM00 = calculateImageMoment(img, 0, 0)
    imgM11 = calculateImageMoment(img, 1, 1)
    imgM02 = calculateImageMoment(img, 0, 2)
    imgM20 = calculateImageMoment(img, 2, 0)
    res = 2 * ((imgM11/imgM00)-(x*y))/(((imgM20/imgM00)-(x*x))-((imgM02/imgM00)-(y*y)))
    res2 = np.arctan(res)
    res2 = res2/2
    return res2


def computeWidthHeight(img, x, y):
    imgM00 = calculateImageMoment(img, 0, 0)
    imgM11 = calculateImageMoment(img, 1, 1)
    imgM02 = calculateImageMoment(img, 0, 2)
    imgM20 = calculateImageMoment(img, 2, 0)
    a = (imgM20/imgM00)-(x**2)
    b = (imgM11/imgM00)-(x*y)
    c = (imgM02/imgM00)-(y**2)
    l = np.sqrt(((a+c) + np.sqrt(b**2+(a-c)**2))/2)
    w = np.sqrt(((a+c) - np.sqrt(b**2+(a-c)**2))/2)
    return l, w


def calculateImageMoment(img_, orderMomentX, orderMomentY):
    img =  cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    imgH = img.shape[0]
    imgW = img.shape[1]
    orderMoment = 0
    for row in range(0, imgH):
        for col in range(0, imgW):
            orderMoment += (col**orderMomentX)*(row**orderMomentY)*img[row,col]
    return orderMoment


def map_intesity_to_bin(intesity_value):
    if intesity_value <= 32:
        return 0
    elif intesity_value <= 64:
        return 1
    elif intesity_value <= 96:
        return 2
    elif intesity_value <= 128:
        return 3
    elif intesity_value <= 160:
        return 4
    elif intesity_value <= 192:
        return 5
    elif intesity_value <= 224:
        return 6
    elif intesity_value <= 256:
        return 7
    
def compute_k_means_filtering(img):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return cv2.normalize(res2,  res2, 0, 255, cv2.NORM_MINMAX)
    


def print_color_histogram(hist):
    for i  in range(0 , hist.shape[0]):
        for j in range(0 , hist.shape[1]):
            for k in range(0, hist.shape[2]):
                print('BIN ', i,j,k,"= ")
                print(hist[i,j,k])

def binarize(img):
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if img[i,j] > 0.01:
                img[i,j] = 255
            else:
                img[i,j] = 0
    return img

def computeWeightedHistogram(img, region_boundaries):
    histogram = np.zeros((8,8 ,8), dtype = float)     
    weight_matrix = np.zeros((img.shape[0],img.shape[1]), np.float)
    origin = np.array([int(math.ceil(region_boundaries["origin_y"])),int(math.ceil(region_boundaries["origin_x"]))])
    regularization_term = 0.0
    for i in range(region_boundaries["from_y"], region_boundaries["to_y"]):
        for j in range(region_boundaries["from_x"], region_boundaries["to_x"]):
            pixel_spatial_position = np.array([i, j])
            weight = compute_gaussian_kernel(origin, pixel_spatial_position)
            i_sub_region = abs(region_boundaries["from_y"] - i)
            j_sub_region = abs(region_boundaries["from_x"] - j)
            weight_matrix[i_sub_region, j_sub_region] = weight
            B = map_intesity_to_bin(img[i, j, 0])
            G = map_intesity_to_bin(img[i, j, 1])
            R = map_intesity_to_bin(img[i, j, 2])
            histogram[B,G,R] += (1 + weight)
            regularization_term += weight
    histogram = (1/regularization_term) * histogram 
    return histogram


def compute_gaussian_kernel(origin, pixel_spatial_position, KERNEL_REG_COEFF = 1.5):
    return math.exp(- (np.linalg.norm((origin - pixel_spatial_position)/KERNEL_REG_COEFF)))


def compare_histograms(hist_a, hist_b):
    for i  in range(0 , hist_a.shape[0]):
        for j in range(0 , hist_a.shape[1]):
            for k in range(0, hist_a.shape[2]):
                if hist_a[i,j,k] != hist_b[i,j,k]:
                    print('distinto')


#Bhattacharyya coefficient
def compute_similarity_coefficient(hist_target, hist_candidate):
    root_product = np.sqrt(hist_target * hist_candidate)
    return root_product.sum()

def map_rgb_to_greyscale(RGB_value):
    B = RGB_value[0]
    G = RGB_value[1]
    R = RGB_value[2]
    return (0.3 * B) + (0.6 * G) + (0.11 * R)

def shift_mass_center(candidate_mass_center, candidate_weight_matrix, region_boundaries):
    from_x = region_boundaries["from_x"]
    to_x = region_boundaries["to_x"]
    from_y = region_boundaries["from_y"]
    to_y = region_boundaries["to_y"]
    denominator = 0.0
    shift_x = 0.0
    shift_y = 0.0
    origin = np.array([candidate_mass_center[1],candidate_mass_center[0]])
    for i in range(from_y, to_y):
        for j in range(from_x, to_x):
            i_sub_img = abs(from_y - i)
            j_sub_img  = abs(from_x - j)
            pixel_spatial_position = np.array([i, j])
            shift_x += i * candidate_weight_matrix[i_sub_img,j_sub_img] * compute_gaussian_kernel(origin, pixel_spatial_position)
            shift_y += j * candidate_weight_matrix[i_sub_img,j_sub_img] * compute_gaussian_kernel(origin, pixel_spatial_position)
            denominator += candidate_weight_matrix[i_sub_img,j_sub_img] * compute_gaussian_kernel(origin, pixel_spatial_position)

    shift_x = shift_x/denominator
    shift_y = shift_y/denominator
    return (shift_y, shift_x)


def compute_weights_matrix(hist_target, hist_candidate, candidate_boundaries, candidate_region):
    height = candidate_region.shape[0]
    width =  candidate_region.shape[1]
    weight_matrix = np.zeros((height, width), np.float)
    for i in range(0, height):
        for j in range(0, width):
            B = map_intesity_to_bin(candidate_region[i, j, 0])
            G = map_intesity_to_bin(candidate_region[i, j, 1])
            R = map_intesity_to_bin(candidate_region[i, j, 2])
            if math.sqrt(hist_target[B,G,R] != 0 and hist_candidate[B,G,R]) != 0:
                weight_matrix[i,j] = math.sqrt(hist_target[B,G,R] / hist_candidate[B,G,R])
            else:
                weight_matrix[i,j] = 0
    return weight_matrix


def print_matrix(matrix):
     for i in range(0, matrix.shape[0]):
        for j in range(0, matrix.shape[1]):
            print(matrix[i,j])

def region_extraction(img, region_boundaries):
    from_x = region_boundaries["from_x"]
    to_x = region_boundaries["to_x"]
    from_y = region_boundaries["from_y"]
    to_y = region_boundaries["to_y"]
    height = abs(from_y - to_y) 
    width =  abs(from_x - to_x) 
    sub_image = np.zeros((height,width,3), int)
    for i in range(from_y, to_y):
        for j in range(from_x, to_x):
            i_sub_img = abs(from_y - i)
            j_sub_img  = abs(from_x - j)
            sub_image[i_sub_img,j_sub_img,0] = img[i,j,0]
            sub_image[i_sub_img,j_sub_img,1] = img[i,j,1]
            sub_image[i_sub_img,j_sub_img,2] = img[i,j,2]
    
    return sub_image


def compute_region_boundaries(blob_params, reference_point):
    central_point_x = reference_point[0]
    central_point_y = reference_point[1]
    MAG_EXPAC_N = 2.5
    MAG_EXPAC_S = 2
    MAG_EXPAC_E = 2.5
    MAG_EXPAC_W = 2
    region_boundaries = {
        "from_x": int(abs(central_point_x - (blob_params["width"] * MAG_EXPAC_W))),
        "to_x": int(abs(central_point_x + (blob_params["width"]* MAG_EXPAC_E))),
        "from_y":  int(abs(central_point_y - (blob_params["longitude"] * MAG_EXPAC_N))),
        "to_y" : int(abs(central_point_y + (blob_params["longitude"] * MAG_EXPAC_S))),
        "origin_x": int(math.ceil(central_point_x)),
        "origin_y": int(math.ceil(central_point_y))
    }
    return region_boundaries


def update_mass_center(centerA, centerB):
    centerA = np.array([centerA[0], centerA[1]])
    centerB = np.array([centerB[0], centerB[1]])
    new_center = (centerA + centerB)/2
    return ((new_center[0]), (new_center[1]))


def computeCentroidDistance(centroidA, centroidB):
    centerA = np.array([centroidA[0], centroidA[1]])
    centerB = np.array([centroidB[0], centroidB[1]])
    return np.linalg.norm(centerA - centerB) 
    

def kernel_track(roi, cap, roi_centroid):

    first_frame =  compute_k_means_filtering(cap[0])
    blob_params = computeBlobParams(roi)
    target_boundaries = compute_region_boundaries(blob_params, roi_centroid)
    probability_map = computeBackProjection(roi, first_frame)
    cv2.imwrite('initial_p-map.png', probability_map)
    target_model = computeWeightedHistogram(probability_map, target_boundaries)
    candidate_boundaries = compute_region_boundaries(blob_params, current_centroid)
    current_centroid = roi_centroid
    epsilon = 0.09
    print('\n-------------------------------------------------------------')
    print('\n\n-------------COMPUTING TRACKING REGISTRY-------------------')
    print('\n\n----------------------------------------------------------\n')
    print('epsilon = ', epsilon)
    input('Press enter to continue...\n')
    tracking_registry = list()

    for i in range(299, 300):  

        frame = compute_k_means_filtering(cap[i])
        probability_map = computeBackProjection(roi, frame)
        cv2.imwrite('initial_p-map.png', probability_map)
        cv2.imwrite('frame.png', frame)
        print('Starting candidate boundaries = ', candidate_boundaries)
        candidate_model = computeWeightedHistogram(probability_map, candidate_boundaries)
        old_similarity_coeff = compute_similarity_coefficient(target_model, candidate_model)
        
        while True: 

            print('\n\nNEW ATTEMPT')
            candidate_region = region_extraction(probability_map, candidate_boundaries)
            weight_matrix = compute_weights_matrix(target_model, candidate_model, candidate_boundaries, candidate_region)
            new_centroid = shift_mass_center(current_centroid, weight_matrix, candidate_boundaries)
            print('NEW CENTROID = ', new_centroid, "ORIGINAL_CENTROID = ", current_centroid)
            candidate_boundaries = update_region_boundaries(candidate_boundaries, new_centroid)
            print('New boundaries = ', candidate_boundaries)
            candidate_model = computeWeightedHistogram(probability_map, candidate_boundaries)
            new_similarity_coeff = compute_similarity_coefficient(target_model, candidate_model)
            print('NSC = ', new_similarity_coeff, ' vs OSC = ', old_similarity_coeff)

            while new_similarity_coeff < old_similarity_coeff:

                print('\n')
                new_centroid = update_mass_center(new_centroid, current_centroid)
                print('updating center = ', new_centroid)
                candidate_boundaries = update_region_boundaries(candidate_boundaries, new_centroid)
                candidate_model = computeWeightedHistogram(probability_map, candidate_boundaries)
                new_similarity_coeff = compute_similarity_coefficient(target_model, candidate_model)
                print('NSC = ', new_similarity_coeff, ' vs OSC = ', old_similarity_coeff)
                print('Frame ' , i)
                print('current candidate boundaries = ' , candidate_boundaries)
                print('\n')

            centroid_distance = computeCentroidDistance(new_centroid, current_centroid)
            print('centroid distance = ', centroid_distance)
            if centroid_distance < epsilon:
                break
            else:
                current_centroid = new_centroid

        print('Frame ' , i)
        tracking_registry.append(candidate_boundaries)
        
    print('\n\ndone.')
    return tracking_registry


def update_region_boundaries(region_boundaries, new_ref_point):
    first_derivative_x = abs(int(math.ceil(new_ref_point[0])) - region_boundaries["origin_x"]) 
    first_derivative_y = abs(int(math.ceil(new_ref_point[1])) - region_boundaries["origin_y"] )
    new_region_boundaries = {
        "from_x":  region_boundaries["from_x"] + origin_x,
        "to_x":  region_boundaries["to_x"] + origin_x,
        "from_y":  region_boundaries["from_y"] + origin_y,
        "to_y" :region_boundaries["to_y"] + origin_y,
        "origin_x": region_boundaries["origin_x"] + first_derivative_x,
        "origin_y": region_boundaries["origin_y"] + first_derivative_y
    }
    return new_region_boundaries
  
    
def loadVideo(path):
    cap = list()
    video = cv2.VideoCapture(path)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0,video_length - 1):
        ret, frame = video.read()
        cap.append(frame)
    video.release()
    return cap


def computeBlobParams(roi):
    image_centroid = calculateImageCentroid(roi)
    longitude, width = computeWidthHeight(roi,image_centroid[0], image_centroid[1])
    orientation = computeOrientation(roi, image_centroid[0], image_centroid[1])
    blob_params  =  {
        "centroidX": int(image_centroid[0]),
        "centroidY": int(image_centroid[1]),
        "longitude": int(longitude),
        "width": int(width),
        "orientation": int(orientation),
    }    
    return blob_params

def paint_rectangle(blob_params, frame):
    cv2.rectangle(frame,(blob_params["from_x"],blob_params["to_y"]),(blob_params["to_x"],blob_params["from_y"]),(0,255,255), 1)
    cv2.imshow('region', frame)
    cv2.waitKey(0)

def show_tracking_registry(cap_path, tracking_registry):
    i = 0
    cap = cv2.VideoCapture(cap_path)
    while(i < len(tracking_registry)):
        blob_params =  tracking_registry[i]
        ret, frame = cap.read()
        cv2.rectangle(frame,(blob_params["from_x"],blob_params["to_y"]),(blob_params["to_x"],blob_params["from_y"]),(0,255,255),3)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
        
    cap.release()
    cv2.destroyAllWindows()


def main():

    cap_path = "slow.mp4"
    roi_centroid = (13,76)
    cap = loadVideo(cap_path)
    roi = cv2.imread('roi.png')
    tracking_registry = kernel_track(roi, cap, roi_centroid)
    input('Press enter to continue...')
    #show_tracking_registry(cap_path, tracking_registry)
    


if __name__ == "__main__":
    main()