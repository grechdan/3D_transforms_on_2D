import cv2
import numpy as np

def corners_calculator(z_r:float, z_s:float, side_length:float,
                      rx:float, ry:float, rz:float,
                      x:float, y:float,
                      res_img:float, center_square:bool) -> list[np.ndarray]:

    # Calculate FOV from side length
    fov = np.degrees(2 * np.arctan(side_length / (2 * z_r)))
    #print("FOV calculated from side length:", fov)

    # Generate corner points based on field of view
    half_fov = np.radians(fov/2)
    #print("half fov:", half_fov)

    #print("Corner points based on field of view:")
    p1 = [-z_r * np.tan(half_fov), z_r * np.tan(half_fov), z_r]
    p2 = [z_r * np.tan(half_fov), z_r * np.tan(half_fov), z_r]
    p3 = [-z_r * np.tan(half_fov), -z_r * np.tan(half_fov), z_r]
    p4 = [z_r * np.tan(half_fov), -z_r * np.tan(half_fov), z_r]
    #print("p1:", p1)
    #print("p2:", p2)
    #print("p3:", p3)
    #print("p4:", p4)

    # Convert angles to radians for rotation
    alpha = np.radians(rx)
    beta = np.radians(ry)
    gamma = np.radians(rz)
    #print("\nAngles to radians for rotation:")
    #print("Alpha:", alpha)
    #print("Beta:", beta)
    #print("Gamma:", gamma)

    # Rotational matrices
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    R = np.matmul(Rz, np.matmul(Ry, Rx))  # Combined rotation matrix
    #print("\nRotational matrices:")
    #print("Rz:")
    #print(Rz)
    #print("Ry:")
    #print(Ry)
    #print("Rx:")
    #print(Rx)
    #print("Combined rotation matrix R:")
    #print(R)

    #print("Corner points with rotation:")
    p1r = np.matmul(R, p1)
    p2r = np.matmul(R, p2)
    p3r = np.matmul(R, p3)
    p4r = np.matmul(R, p4)
    #print("p1r:", p1r)
    #print("p2r:", p2r)
    #print("p3r:", p3r)
    #print("p4r:", p4r)

    # create line equations from origin and rotated corner points
    # lines will be treated as vectors. Vector equation is: r(P) = r(P1) + lambda*(r(P2)-r(P1))
    # determine lambda as z_s/z_r as a scaling factor
    lambda1 = z_s / p1r[2]
    lambda2 = z_s / p2r[2]
    lambda3 = z_s / p3r[2]
    lambda4 = z_s / p4r[2]

    # using P1 as origin to make equation simpler: r(P) = [0,0,0] + lambda*r(P2) = lambda*r(P2)
    p1s = lambda1 * p1r
    p2s = lambda2 * p2r
    p3s = lambda3 * p3r
    p4s = lambda4 * p4r

    # Translate the corner points
    #print("translate corner points:")
    p1s = np.array([p1s[0] + x, p1s[1] + y, p1s[2]]) / res_img
    p2s = np.array([p2s[0] + x, p2s[1] + y, p2s[2]]) / res_img
    p3s = np.array([p3s[0] + x, p3s[1] + y, p3s[2]]) / res_img
    p4s = np.array([p4s[0] + x, p4s[1] + y, p4s[2]]) / res_img
    #print("p1s:", p1s)
    #print("p2s:", p2s)
    #print("p3s:", p3s)
    #print("p4s:", p4s)

    # Assign scaled points to square corner positions
    corners = [p1s[0:2], p2s[0:2], p3s[0:2], p4s[0:2]]
    #print("p1s[0:2]:","p2s[0:2]:","p3s[0:2]:","p4s[0:2]:", p1s, p2s,p3s,p4s)

    # Center square to image center based on its centroid if required
    if center_square:
        centroid = [np.mean([corners[i][0] for i in range(0, len(corners), 1)]),
                    np.mean([corners[i][1] for i in range(0, len(corners), 1)])]
        corners = [corners[i] - centroid for i in range(0, len(corners), 1)]

    return corners

def find_intersection(p1:np.ndarray, p2:np.ndarray, p3:np.ndarray, p4:np.ndarray) -> np.ndarray:

    A1 = p2[1] - p1[1]
    B1 = p1[0] - p2[0]
    C1 = A1 * p1[0] + B1 * p1[1]

    A2 = p4[1] - p3[1]
    B2 = p3[0] - p4[0]
    C2 = A2 * p3[0] + B2 * p3[1]

    determinant = A1 * B2 - A2 * B1
    x = (B2 * C1 - B1 * C2) / determinant
    y = (A1 * C2 - A2 * C1) / determinant

    return np.array([x, y], dtype=np.int32)


def checkerboard_generator(dim_img: tuple, num:int, 
                           z_r:float, z_s:float, side_length:float, 
                           rx:float = 0, ry:float = 0, rz:float = 0,
                           x:float = 0, y:float = 0,
                           res_img:float = 1, gen_img:bool = True, center_square:bool = False):
    
    # Generate the image
    
    thickness = 5
    
    if gen_img:
    
        image = np.zeros((dim_img[1], dim_img[0]), dtype="uint8")
    
        corners = corners_calculator(z_r, z_s, side_length, rx, ry, rz, x, y, res_img, center_square)
        
        corners_img = [[int(dim_img[0] / 2 + corners[i][0]), int(dim_img[1] / 2 - corners[i][1])] for i in
                       range(0, len(corners), 1)]
        
        for i in range(0, len(corners_img), 1):
            
            if corners_img[i][0] < 0:
                
                corners_img[i][0] = 0
            
            elif corners_img[i][0] >= dim_img[0]:
                
                corners_img[i][0] = dim_img[0] - 1
            
            if corners_img[i][1] < 0:
                
                corners_img[i][1] = 0
            
            elif corners_img[i][1] >= dim_img[1]:
                
                corners_img[i][1] = dim_img[1] - 1

        cb_coordinates = np.empty((num + 1, num + 1), dtype=object)

        cb_coordinates[0][0] = np.array(corners_img[2], dtype=np.int32)
        cb_coordinates[0][num] = np.array(corners_img[3], dtype=np.int32)
        cb_coordinates[num][0] = np.array(corners_img[0], dtype=np.int32)
        cb_coordinates[num][num] = np.array(corners_img[1], dtype=np.int32)

        for i in range(num + 1):

            for j in range(num + 1):

                if i == 0:

                    if j != 0 and j != num:

                        cb_coordinates[i][j] = cb_coordinates[0][0] + (cb_coordinates[0][num] - cb_coordinates[0][0]) * j / num

                elif i == num:

                    if j != 0 and j != num:

                        cb_coordinates[i][j] = cb_coordinates[num][0] + (cb_coordinates[num][num] - cb_coordinates[num][0]) * j / num

                else:

                    if j == 0:
        
                        cb_coordinates[i][j] = cb_coordinates[0][0] - (cb_coordinates[0][0] - cb_coordinates[num][0]) * i / num


                    elif j == num:
        
                        cb_coordinates[i][j] = cb_coordinates[0][num] - (cb_coordinates[0][num] - cb_coordinates[num][num]) * i / num

                    else:

                        cb_coordinates[i][j] = find_intersection(cb_coordinates[0][j], cb_coordinates[num][0] + (cb_coordinates[num][num] - cb_coordinates[num][0]) * j / num, cb_coordinates[i][0], 
                                                                 cb_coordinates[0][num] - (cb_coordinates[0][num] - cb_coordinates[num][0]) * i / num)
        
        print(cb_coordinates)
 
        for i in range(num):

            for j in range(num):

                if (i + j) % 2 == 0:

                    color = (255, 255, 255)                     
                
                else: 

                    color = (0, 0, 0)



                points = np.array([cb_coordinates[i+1][j], cb_coordinates[i+1][j+1], cb_coordinates[i][j+1], cb_coordinates[i][j]], dtype=np.int32)
                points = points.reshape((-1, 1, 2))

                image = cv2.fillPoly(image, [points], color)
        
        image = cv2.line(image, corners_img[0], corners_img[1], color, thickness)
        image = cv2.line(image, corners_img[0], corners_img[2], color, thickness)
        image = cv2.line(image, corners_img[2], corners_img[3], color, thickness)
        image = cv2.line(image, corners_img[3], corners_img[1], color, thickness)

    else:
        
        image = []

    return image


def main():

        # Parameters for square generation
    num = 5 # 
    z_r = 100 # Position of the generated square. Unit is [mm].
    z_s = 100 # Position of the screen where the square will be viewed. Unit is [mm].
    side_length = 1000  # Side length of the square. Unit is [mm].
    # rx = 0 # Rotation of the square around the x-axis. Unit is [°].
    # ry = 0 # Rotation of the square around the y-axis. Unit is [°].
    # rz = 0  # Rotation of the square around the z-axis. Unit is [°].
    # x = 0 # Translation of the square along the x-axis. Unit is [mm].
    # y = 0 # Translation of the square along the y-axis. Unit is [mm].
    # z = 0 # translation of the rectangle along the z-axis. Unit is [mm] - not used !!!
    # res_img = 1 #Resolution of one pixel. Unit is [mm/pixel].
    dim_img = (10000,8000) # Resolution of the image that will be generated from the screen. Provided as (horizontal, vertical).[pixels^2]
    # center_square = False - Flag to center square to image center based on its centroid.
    # gen_img = True - Flag to set if image should be generated or not.

    image = checkerboard_generator(dim_img, num, z_r, z_s, side_length)

    # Save the image
    cv2.imwrite("checkerboard.jpg", image)


    # Save the corner points, x, y, rx, ry, and rz in a text file
    #with open("corner_points_and_params.txt", "w") as file:

        #file.write(f"p1s: {corners[0]}, p2s: {corners[1]}, p3s: {corners[2]}, p4s: {corners[3]}\n")
        #file.write(f"x: {x}, y: {y}, rx: {rx}, ry: {ry}, rz: {rz}")

    # Display the image
    #cv2.imshow("Square Image", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

main()
