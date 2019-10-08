import cv2
import numpy

import numpy as np
import matplotlib.pyplot as plt


def same_point_acquisition(img):
    def draw_dot(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
    points = []
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('img', draw_dot)
    while True:
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
    return np.array(points)


def affine_transformation_model_solve(goal_point, trans_point):
    u_mat = np.mat(goal_point)
    one = np.ones(trans_point.shape[0])
    x_mat = np.mat(np.insert(trans_point, 2, values=one, axis=1))
    affine_mat = np.mat(np.linalg.inv(x_mat.T * x_mat) * x_mat.T * u_mat)
    # affine_mat = np.mat(np.insert(affine_mat_pre.T, 2, values=[0, 0, 1], axis=0))
    return affine_mat


def nearest_neighbour(image, coord: tuple):
    if 0 <= round(coord[0]) < image.shape[0] and 0 <= round(coord[1]) < image.shape[1]:
        return image[int(round(coord[0])), int(round(coord[1]))]
    return 1


def bilinear_interpolation(image, coord: tuple):
    x = (int(coord[0]), coord[0] - int(coord[0]))
    y = (int(coord[1]), coord[1] - int(coord[1]))
    a = np.array([[1 - y[1] , y[1]]])
    if len(image.shape) == 2:
        b = np.zeros((2, 2))
    else:
        b = np.zeros((2, 2, image.shape[-1]))
    for index1 in range(b.shape[0]):
        for index2 in range(b.shape[1]):
            b[index1, index2] = nearest_neighbour(image, (y[0] + index1, x[0] + index2))
    c = np.array([[1 - x[1], x[1]]])
    if len(image.shape) == 2:
        return np.dot(np.dot(a, b), c.T)
    else:
        colors = []
        for sub in range(image.shape[-1]):
            colors.append(np.dot(np.dot(a, b[..., sub]), c.T))
        return colors


image_a = cv2.imread('./klcc_a.png', cv2.IMREAD_COLOR)
image_b = cv2.imread('./klcc_b.png', cv2.IMREAD_COLOR)

points_a = same_point_acquisition(image_a)
points_b = same_point_acquisition(image_b)
# points_a = np.array([[1292, 309], [1377, 1405], [1422, 1618]])
# points_b = np.array([[963, 384], [1084, 1480], [1142, 1693]])

affine_mat = affine_transformation_model_solve(points_a, points_b)
anti_affine_mat = affine_transformation_model_solve(points_b, points_a)

x_max = image_a.shape[1] - 1
for y in range(0, image_b.shape[0],  image_b.shape[0] - 1):
    for x in range(0, image_b.shape[1],  image_b.shape[1] - 1):
        goal_point = np.mat([x, y, 1]) * affine_mat
        if round(goal_point[0, 0]) > x_max:
            x_max = int(round(goal_point[0, 0]))

new_image = np.zeros((image_a.shape[0], x_max, image_a.shape[-1]), dtype=np.uint8)
new_image[:, :image_a.shape[1]] = image_a
for y in range(new_image.shape[0]):
    for x in range(image_a.shape[1], new_image.shape[1]):
        origin_point = np.mat([[x, y, 1]])
        goal_point = origin_point * anti_affine_mat
        new_image[y, x] = bilinear_interpolation(image_b, tuple(list(goal_point.A)[0]))
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', new_image)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite('./klcc.png', new_image)