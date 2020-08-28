import cv2
import numpy as np
from matplotlib import pyplot as plt


drawing = False # true if mouse is pressed
oldx , oldy = None , None

# mouse callback function
# TODO: draw ticker line when the mouse is moving slower. Need to record event time...
def line_drawing(event, curx, cury, flags, img):
    global oldx, oldy, drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        oldx, oldy = curx, cury
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img, (oldx, oldy), (curx, cury), color=255, thickness=24)
            oldx, oldy = curx, cury
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img, (oldx, oldy), (curx, cury), color=255, thickness=24)


def draw(sx, sy, title='Enter to save. Esc to close', smooth=True):
    img = np.full((sx, sy, 1), 0, dtype=np.uint8)
    # Draw with the mouse
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, line_drawing, img)
    key = None
    while True:
        cv2.imshow(title, img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 13:
            break
    cv2.destroyAllWindows()

    if key == 27:
        return None
    else:
        if smooth:
            img = cv2.GaussianBlur(img, (5, 5), 0)
        return img


if __name__ == "__main__":
    image = draw(384, 384)
    if image is not None:
        plt.imshow(image, cmap='gray')  # , cmap="Greys"
        plt.show()
        cv2.imwrite("drawing.png", image)
