
import numpy as np
import cv2 
import math



def computeBackProjection(roi, target):
    roi_hist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
    channels = [0,1,2]
    binsBGR = [8,8,8]
    ranges = [0,256,0,256,0,256]
    bgrhist = cv2.calcHist([roi],channels, None, binsBGR, ranges )
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)
    ret,thresh = cv2.threshold(dst,50,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(target,thresh)
    res = np.vstack((target,thresh,res))
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
    l = math.sqrt(((a+c) + math.sqrt(b**2+(a-c)**2))/2)
    w = math.sqrt(((a+c) - math.sqrt(b**2+(a-c)**2))/2)
    return l, w


def calculateImageMoment(img, orderMomentX, orderMomentY):
    imgH = img.shape[0]
    imgW = img.shape[1]
    orderMoment = 0
    for row in range(0, imgH):
        for col in range(0, imgW):
            orderMoment += (col**orderMomentX)*(row**orderMomentY)*img[row][col]
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
    
def compute_k_means_filtering(frame):
    pass


def computeWeightedHistogram(img, region_boundaries):
    KERNEL_REG_TERM = 1.5
    histogram = np.zeros((8,8 ,8), dtype = int) 
    weight_matrix = np.zeros((height,width,3), np.float)
    origin = np.array([region_boundaries["origin_x"],region_boundaries["origin_y"]])
    regularization_term = 0.0
    for i in range(region_boundaries["from_x"], region_boundaries["to_x"]):
        for j in range(region_boundaries["from_y"], region_boundaries["to_y"]):
            pixel_spatial_position = np.array([i, j])
            weight = compute_gaussian_kernel(origin, pixel_spatial_position, KERNEL_REG_TERM)
            i_sub_region = region_boundaries["from_x"] - i
            j_sub_region = region_boundaries["from_y"] - j
            weight_matrix[i_sub_region, j_sub_region] = weight
            B = map_intesity_to_bin(img[i, j, 0])
            G = map_intesity_to_bin(img[i, j, 1])
            R = map_intesity_to_bin(img[i, j, 2])
            histogram[B,G,R] = histogram[B,G,R] + (1 + weight)
            regularization_term += weight
    histogram = 1/regularization_term * histogram 
    cv.imwrite('weight matrix.png', weight_matrix)
    return histogram


def compute_gaussian_kernel(origin, pixel_spatial_position, KERNEL_REG_TERM = 0.5):
    return math.exp(- (np.linalg.norm((origin - pixel_spatial_position)/KERNEL_REG_TERM)))


#Bhattacharyya coefficient
def compute_similarity_coefficient(hist_target, hist_candidate):
    root_product = math.sqrt(hist_target * hist_candidate)
    return root_product.sum()


def shift_mass_center(hist_target, hist_candidate, candidate_mass_center, candidate_weight_matrix, candidate_region):
    width = candidate_= region.shape[0]
    height = candidate_region.shape[1]
    numerator = 0.0
    denominator = 0.0
    origin = np.array([candidate_mass_center[0],candidate_mass_center[1]])
    for i in range(0, width):
        for j in range(0, height):
            pixel_spatial_position = np.array([i, j])
            numerator += candidate_region[i,j] * candidate_weight_matrix[i,j] * compute_gaussian_kernel(origin, pixel_spatial_position)
            denominator += candidate_weight_matrix[i,j] * compute_gaussian_kernel(origin, pixel_spatial_position)
    mass_center = numerator/denominator
    return (candidate_mass_center[0] + mass_center, candidate_mass_center[1] + mass_center)


def compute_weights_matrix(hist_target, hist_candidate, candidate_boundaries, candidate_region):
    height = candidate_region.shape[0]
    width =  candidate_region.shape[1]
    weight_matrix = np.array((height, width), np.float)
    for i in range(0, width):
        for j in range(0, height):
            B = map_intesity_to_bin(candidate_region[i, j, 0])
            G = map_intesity_to_bin(candidate_region[i, j, 1])
            R = map_intesity_to_bin(candidate_region[i, j, 2])
            weight_matrix[i,j] = math.sqrt(hist_target[B,G,R] / hist_candidate[B,G,R])
    return weight_matrix


def region_extraction(img, region_boundaries):
    from_x = region_boundaries["from_x"]
    to_x = region_boundaries["to_x"]
    from_y = region_boundaries["from_y"]
    to_y = region_boundaries["to_y"]
    height = abs(from_y - to_y) 
    width =  abs(from_x - to_x) 
    sub_image = np.zeros((height,width,3), np.uint8)
    k = 0
    l = 0
    for i in range(from_x, to_x):
        l = 0
        for j in range(from_y, to_y):
            sub_image.itemset((l,k,0),img[j,i,0])
            sub_image.itemset((l,k,1),img[j,i,1])
            sub_image.itemset((l,k,2),img[j,i,2])
            l += 1
        k += 1
    return sub_image


def compute_region_boundaries(blob_params, reference_point):
    central_point_x = reference_point[0]
    central_point_y = reference_point[1]
    region_boundaries = {
        "from_x": abs(central_point_x - blob_params["longitude"]/2),
        "to_x": abs(central_point_x + blob_params["longitude"]/2),
        "from_y":  abs(central_point_y - blob_params["width"]/2),
        "to_y" : abs(central_point_y + blob_params["width"]/2),
        "origin_x": central_point_x,
        "origin_y": central_point_y
    }
    return region_boundaries


def update_mass_center(centerA, centerB):
    centerA = np.array([centerA[0], centerA[1]])
    centerB = np.array([centerB[0], centerB[1]])
    new_center = (centerA + centerB)/2
    return (new_center[0], new_center[1])


def computeCentroidDistance(centroidA, centroidB):
    centerA = np.array([centroidA[0], centroidA[1]])
    centerB = np.array([centroidB[0], centroidB[1]])
    return np.linalg.norm(centerA - centerB) 
    

def kernel_track(roi, cap, roi_centroid):

    first_frame = compute_k_means_filtering(cap[0])
    roi = compute_k_means_filtering(roi)
    blob_params = computeBlobParams(roi)
    probability_map = computeBackProjection(roi, first_frame)
    target_model = computeWeightedHistogram(probability_map, target_boundaries)
    current_centroid = roi_centroid
    candidate_boundaries = compute_region_boundaries(blob_params, current_centroid)
    candidate_region = region_extraction(probability_map, candidate_boundaries)
    epsilon = 0.01
    print('\n-------------------------------------------------------------')
    print('\n\n-------------COMPUTING TRACKING REGISTRY-------------------')
    print('\n\n----------------------------------------------------------\n')
    print('epsilon = ', epsilon)
    input('Press enter to continue...\n')
    tracking_registry = list()
    num_frame = 0

    for frame in cap:  

        frame = compute_k_means_filtering(frame)
        probability_map = computeBackProjection(roi, frame)
        candidate_model = computeWeightedHistogram(probability_map, candidate_boundaries)
        old_similarity_coeff = compute_similarity_coefficient(target_model, candidate_model)

        while centroid_distance > epsilon: 

            weight_matrix = compute_weights_matrix(target_model, candidate_model, candidate_boundaries)
            new_centroid = shift_mass_center(target_model, candidate_model, current_centroid, weight_matrix, candidate_region)
            candidate_boundaries = compute_region_boundaries(blob_params, new_centroid)
            candidate_model = computeWeightedHistogram(probability_map, candidate_boundaries)
            new_similarity_coeff = compute_similarity_coefficient(target_model, candidate_model)
           
            while new_similarity_coeff < old_similarity_coeff:

                print('Updating mass center...')
                new_centroid = update_mass_center(new_centroid, current_centroid)
                candidate_boundaries = compute_region_boundaries(blob_params, new_centroid)
                candidate_model = computeWeightedHistogram(probability_map, candidate_boundaries)
                new_similarity_coeff = compute_similarity_coefficient(target_model, candidate_model)
            
            centroid_distance = computeCentroidDistance(new_centroid, current_centroid)
            if centroid_distance < epsilon:
                break
            else:
                current_centroid = new_centroid

        num_frame += 1
        print('Frame ' , num_frame)
        tracking_registry.append(candidate_boundaries)
        
    print('\n\ndone.')
    return tracking_registry


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
    longitude, width = computeWidthHeight(roi,image_centroid[0], calculateImageCentroid[1])
    orientation = computeOrientation(roi, imag_centroid[0], image_centroid[1])
    blob_params  =  {
        "centroidX": image_centroid[0],
        "centroidY": image_centroid[1],
        "longitude": longitude,
        "width": width,
        "orientation": orientation,
    }    
    return blob_params

def show_tracking_registry(cap_path, tracking_registry):
    cap = cv2.VideoCapture('wow.avi')
    i = 0
    while(cap.isOpened()):
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
    cap_path = "highway_460to869.avi"
    roi_centroid = (100,100)
    roi = cv2.imread('')
    cap = loadVideo(cap_path)
    tracking_registry = kernel_track(roi, cap, roi_centroid)
    show_tracking_registry(cap_path, tracking_registry)



if __name__ == "__main__":
    main()