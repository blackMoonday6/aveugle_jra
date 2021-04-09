## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
from __future__ import division
import pyrealsense2 as rs
from models import *
from utils.datasets import *
from util import *
import cv2
import pickle as pkl
import tkinter
from PIL import Image, ImageTk
import os
from os import path
import time
import texteditor

def readLab():
    f = open("custom/classes.names", "r+")
    fl = f.readlines()
    for x in fl:
        name = x
        print(name)

def OpenReadme():
    text = texteditor.open(filename='README.md')

def capture():
    global count
    global depth_colormap
    global color_image
    global cap
    global button_widget
    global button_widget1
    global button_widget2
    global window
    global canvas2
    global listbox
    button_widget4.pack_forget()
    button_widget5.pack_forget()
    button_widget6.pack_forget()
    print('hey')
    if (path.exists('custom') == False):
        os.mkdir('custom')
    if (path.exists('custom/classes.names') == False):
        f = open("custom/classes.names", "w+")
    readLab()
    color_image = Image.fromarray(color_image)
    depth_colormap = Image.fromarray(depth_colormap)
    if (path.exists('custom/depth') == False):
        os.mkdir('custom/depth')
    if (path.exists('custom/images') == False):
        os.mkdir('custom/images')
    else:
        count = len(os.listdir('custom/images'))
    if (path.exists("custom/images/color" + str(count) + ".jpg") == False):
        color_image.save("custom/images/color" + str(count) + ".jpg")
    if (path.exists("custom/depth/stereo" + str(count) + ".jpg") == False):
        depth_colormap.save("custom/depth/stereo" + str(count) + ".jpg")
    count = count + 1
    button_widget = tkinter.Button(window, text="add", command=Add)
    button_widget1 = tkinter.Button(window, text="delete", command=Delete)
    button_widget2 = tkinter.Button(window, text="save", command=save)
    cap = False
    listbox = tkinter.Listbox(window)
    names = open("custom/classes.names", "r")
    for line in names:
        line = line.replace('\n', '')
        listbox.insert(tkinter.END, line)
    return count


def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def on_button_press(event):
    global window
    global start_x
    start_x = event.x
    global start_y
    start_y = event.y
    global rect
    # create rectangle if not yet exist
    rect = window.canvas.create_rectangle(start_x, start_y, 1, 1, outline='red')


def back():
    global count
    global cap
    global button_widget
    global button_widget1
    global button_widget2
    global button_widget4
    global listbox
    print(cap)
    if cap:
        count = capture()
    else:
        button_widget4.pack()
        button_widget5.pack()
        button_widget6.pack()
        button_widget.pack_forget()
        button_widget1.pack_forget()
        button_widget2.pack_forget()
        listbox.pack_forget()
        cap = True


def on_move_press(event):
    global window
    global rect
    global start_x
    global start_y
    curX = event.x
    curY = event.y
    # # expand rectangle as you drag the mouse
    window.canvas.coords(rect, start_x, start_y, curX, curY)


def on_button_release(event):
    global window
    global rect
    global start_x
    global start_y
    global curX
    global curY
    global cord
    curX = event.x
    curY = event.y
    window.canvas.coords(rect, start_x, start_y, curX, curY)
    centx = str((((curX - start_x) / 2) + start_x) / 640)
    centy = str((((curY - start_y) / 2) + start_y) / 480)
    width = str(abs(curX - start_x) / 640)
    height = str(abs(curY - start_y) / 480)
    cord = centx + ' ' + centy + ' ' + width + ' ' + height
    print(start_x - curX, start_y - curY)
    # comm = tkinter.Label(window, text="Label me!").grid(row=3)
    # labelList = tkinter.Listbox(window)


def Add():
    global lab
    print('saved!')
    commWin = tkinter.Toplevel(window)
    commWin.title("add a new label")
    comm = tkinter.Label(commWin, text="please enter the name of the label to be added!").pack()
    lab = tkinter.Entry(commWin)
    lab.pack()
    but = tkinter.Button(commWin, text="add to the list", command=inser).pack()


def save():
    global cord
    global listbox
    global cap
    # print('rect saved')
    listbox.curselection()
    if (path.exists('custom/labels') == False):
        os.mkdir('custom/labels')
    if path.exists("custom/labels/color" + str(count - 1) + str(count) + ".txt") == False:
        f = open("custom/labels/color" + str(count - 1) + ".txt", "a")
        f.write(str(listbox.curselection()[0]))
        f.write(' ')
        f.write(str(cord))
        back()
        cap = True
        f.write(str('\n'))


def Delete():
    global listbox
    selection = listbox.curselection()
    listbox.delete(selection[::-1])
    # commWin = tkinter.Toplevel(window)
    # commWin.title("Label")
    # comm = tkinter.Label(window, text="Label me!").grid(row=3)
    # # labelList = tkinter.Listbox(window)
    # tkinter.Checkbutton(window, text="colored plastique").grid(columnspan=2)
    # tkinter.Checkbutton(window, text="colored plastique").grid(columnspan=2)


def inser():
    global lab
    print('added to the list')
    name = str(lab.get())
    listbox.insert(tkinter.END, name)
    f = open("custom/classes.names", "a")
    f.write(name)
    f.write('\n')
    f.close()
    lab.delete(0, tkinter.END)


def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    labeling = "{0}".format(classes[cls]) + " " + str(round(x[6].item() * 100, 2)) + " %"
    conf = str(x[6])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(labeling, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 1.5, c1[1] + t_size[1] + 2
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, labeling, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 0.7, [225, 255, 255], 1);
    return img


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)
count = 0
start_x = None
start_y = None
cap = True
rect = None
# Start streaming

cfgfile = "config/yolov3.cfg"
weightsfile = "weights/yolov3.weights"
num_classes = 80

confidence = float(0.5)
nms_thesh = float(0.1)
start = 0

bbox_attrs = 5 + num_classes
device = torch.device("cuda")
model = Darknet(cfgfile, 470).to(device)
classes = load_classes('data/coco.names')
Tensor = torch.cuda.FloatTensor
if weightsfile.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(weightsfile)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(weightsfile))

inp_dim = 480

cord = ''
assert inp_dim % 32 == 0
assert inp_dim > 32

window = tkinter.Tk()
# to rename the title of the window
window.title("Dataset builder")


button_widget7 = tkinter.Button(window, text="Documentation", command=OpenReadme)
button_widget7.pack()
window.canvas = tkinter.Canvas(width=1280, height=480)
window.canvas.pack()
button_widget4 = tkinter.Button(window, text="Build Data", command=capture)
button_widget5 = tkinter.Button(window, text="Train", command=capture)
button_widget6 = tkinter.Button(window, text="Test", command=capture)
button_widget4.pack()
button_widget5.pack()
button_widget6.pack()
align_to = rs.stream.color
align = rs.align(align_to)
# canvas2 = tkinter.Canvas(width=1280, height=20)
# canvas2.grid(row=1)
try:
    while True:
        start_time = time.time()
        # Wait for a coherent pair of frames: depth and color
        if cap == True:
            frames = pipeline.wait_for_frames(10000)
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())



            img = prep_image(color_image, inp_dim)
            #
            output = model(Variable(img[0].type(Tensor)))
            output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
            #
            output[:, 1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim)) / inp_dim
            output[:, [1, 3]] *= color_image.shape[1]
            output[:, [2, 4]] *= color_image.shape[0]

            #
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("palette", "rb"))
            #list(map(lambda x: write(x, color_image), output))

            #cv2.rectangle(color_image, (output[0][1], y), (x + w, y + h), (0, 255, 0), 2)
            d = round(1 / (time.time() - start_time), 2)
            if(output.size()[0]>0):
                for i in range(output.size()[0]):
                    w = output[i][1]+(output[i][3]-output[i][1])/2
                    h = output[i][2]+(output[i][4]-output[i][2])/2
                    #cv2.putText(color_image, str(round(x.item(),2)) + ',' + str(round(y.item(),2)),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
                    #print(aligned_depth_frame.get_distance(int(x), int(y)))
                    (x,y,z)=(rs.rs2_deproject_pixel_to_point(depth_intrin, [int(w), int(h)], aligned_depth_frame.get_distance(int(w), int(h))))
                    print("{0} is in : (x={1} , y={2} , z={3}) at {4} fps".format(classes[int(output[i][-1])],round(x,2),round(y,2),round(z,2),d))
            # Stack both images horizontally
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(2700-depth_image, alpha=0.1), cv2.COLORMAP_JET)
            list(map(lambda x: write(x, depth_colormap), output))
            depth_colormap = cv2.resize(depth_colormap, (640, 480))
            images = np.hstack((color_image, depth_colormap))
            images = Image.fromarray(images)
            photo = ImageTk.PhotoImage(images)
            window.canvas.create_image((640, 240), image=photo)

        else:
            button_widget.pack()
            button_widget1.pack()
            button_widget2.pack()
            listbox.pack()
            window.canvas.bind("<ButtonPress-1>", on_button_press)
            window.canvas.bind("<B1-Motion>", on_move_press)
            window.canvas.bind("<ButtonRelease-1>", on_button_release)


        window.update()
except Exception as e:
    print(e)
    pass

# finally:

# Stop streaming
# pipeline.stop()