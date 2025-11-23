import cv2
import numpy as np
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# 3D Cube vertices and edges
def Cube():
    vertices = [
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, 1],
        [1, -1, 1],
        [-1, -1, 1],
        [-1, 1, 1]
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

# Hand gesture processing
def get_hand_gestures(results, w, h):
    """
    Returns:
        rotate_x, rotate_y: rotation deltas
        scale: scale factor
    """
    rotate_x, rotate_y = 0, 0
    scale = 1.0
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        # Get wrist and index tip
        wrist = hand_landmarks.landmark[0]
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        # Map hand movement to rotation
        hand_x = int(index_tip.x * w)
        hand_y = int(index_tip.y * h)
        # Use static variables to store previous position
        if (not hasattr(get_hand_gestures, 'prev_x') or
            get_hand_gestures.prev_x is None or
            get_hand_gestures.prev_y is None):
            get_hand_gestures.prev_x = hand_x
            get_hand_gestures.prev_y = hand_y
            rotate_x = 0
            rotate_y = 0
        else:
            rotate_x = (hand_y - get_hand_gestures.prev_y) * 0.5
            rotate_y = (hand_x - get_hand_gestures.prev_x) * 0.5
            get_hand_gestures.prev_x = hand_x
            get_hand_gestures.prev_y = hand_y
        # Pinch to scale
        pinch_dist = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
        scale = np.clip(2 * pinch_dist, 0.5, 2.0)
    else:
        # Reset previous position if no hand
        get_hand_gestures.prev_x = None
        get_hand_gestures.prev_y = None
    return rotate_x, rotate_y, scale

def main():
    # openCV camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    # pediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    # pygame + opengl setup
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    # state
    rot_x, rot_y = 0, 0
    scale = 1.0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    # cam frame
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        h, w, _ = frame.shape
    # gestures
        d_rot_x, d_rot_y, new_scale = get_hand_gestures(results, w, h)

    # scale updating
        if results.multi_hand_landmarks:
            rot_x += d_rot_x
            rot_y += d_rot_y
            scale = new_scale

        rot_x = np.clip(rot_x, -90, 90)
        rot_y = np.clip(rot_y, -90, 90)
        scale = np.clip(scale, 0.5, 2.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glScalef(scale, scale, scale)
        glRotatef(rot_x, 1, 0, 0)
        glRotatef(rot_y, 0, 1, 0)
        Cube()
        glPopMatrix()
        pygame.display.flip()
        pygame.time.wait(10)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow(':)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
    cap.release()
    hands.close()
    pygame.quit()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 