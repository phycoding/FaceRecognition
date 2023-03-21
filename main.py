from io import BytesIO
import streamlit as st
import numpy as np
import mediapipe as mp
import cv2
import tempfile
from deepface import DeepFace

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

from PIL import Image, ImageColor
#from streamlit_webrtc import webrtc_streamer, RTCConfiguration
#import av
import cv2



im1 = st.sidebar.file_uploader("Upload Image (Only jpg and png files are allowed)", type=['jpg', 'png'])
if im1 is not None:
    im1 = cv2.imdecode(np.fromstring(im1.read(), np.uint8), 1)
# Set page configs. Get emoji names from WebFx
#st.set_page_config(page_title="Real-time Face Detection")

# -------------Header Section------------------------------------------------

title = '<p style="text-align: center;font-size: 40px;font-weight: 550; ">  Real-time Monitoring System</p>'
st.markdown(title, unsafe_allow_html=True)

st.markdown(
    "Live camera analysis using state-of-the-art DeepLearning Algorithm"
    )

supported_modes = "<html> " \
                  "<body><div> <b>Supported Face Detection Modes (Change modes from sidebar menu)</b>" \
                  "<ul><li>CCTV Video Upload</li><li>Live CCTV</li></ul>" \
                  "</div></body></html>"
st.markdown(supported_modes, unsafe_allow_html=True)

st.warning("NOTE : Click the arrow icon at Top-Left to open Sidebar menu. ")

# -------------Sidebar Section------------------------------------------------

detection_mode = None
# Haar-Cascade Parameters
minimum_neighbors = 4
# Minimum possible object size
min_object_size = (50, 50)
# bounding box thickness
bbox_thickness = 3
# bounding box color
bbox_color = (0, 255, 0)

with st.sidebar:
    #st.image("./assets/faceman_cropped.png", width=260)

    title = '<p style="font-size: 25px;font-weight: 550;">Face Detection Settings</p>'
    st.markdown(title, unsafe_allow_html=True)

    # choose the mode for detection
    mode = st.radio("Choose Face Detection Mode", ('CCTV Video Upload',
                                                   'Live CCTV'
                                                   ), index=0)
    if mode == 'CCTV Video Upload':
        detection_mode = mode
    elif mode == 'Live CCTV':
        detection_mode = mode

    # slider for choosing parameter values
    Model = st.selectbox("Choose your Model",
        ("ir18", "ir50")
    )

    

    # Get bbox color and convert from hex to rgb
    bbox_color = ImageColor.getcolor(str(st.color_picker(label="Bounding Box Color", value="#00FF00")), "RGB")

    # ste bbox thickness
    bbox_thickness = st.slider("Bounding Box Thickness", min_value=1, max_value=30,
                               help="Sets the thickness of bounding boxes",
                               value=bbox_thickness)

    st.info("NOTE : The quality of detection will depend on Model chosen"
            )

# -------------Image Upload Section------------------------------------------------


if detection_mode == "CCTV Video Upload":

    uploaded_file = st.file_uploader("Upload Video (Only mp4 and mkv files are allowed)", type=['.mp4', '.mkv'])
    tfile = tempfile.NamedTemporaryFile(delete=False)

    if uploaded_file is not None:
        tfile.write(uploaded_file.read())

        with st.spinner("Finding the person by analyzing video..."):
            cap = cv2.VideoCapture(tfile.name)
            while True:
                ret, frame = cap.read()
                # Convert the frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                results = face_detection.process(frame)
                # To convert PIL Image to numpy array:
                # For each face, crop the face out of the frame using the bounding box coordinates
                if results.detections:

                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, width, height = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)
                        face = frame[y:y+height, x:x+width]
                        result = DeepFace.verify(face, im1,enforce_detection=False)
                        if result >= 0.4:
                            st.title("Person Found")

            cap.release()



# -------------Webcam Image Capture Section------------------------------------------------


# -------------Hide Streamlit Watermark------------------------------------------------
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)