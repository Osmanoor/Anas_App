import streamlit as st
import tempfile
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import pipeline
import crop

# Set the title of the app
st.title("Visual Speech Recognition")

# Add a header
st.header("Upload your MP4 video file")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an MP4 video", type=["mp4"])

if uploaded_file is not None:
    # Display the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_file = temp_file.name
else:
    video_file = "download.mp4"
st.video(video_file)

#Predict The Text
result = pipeline.VSR(video_file)
st.text("You said: " + result)

#Crop video
crop_video = "crop_video.mp4"
crop.preprocess_video(video_file,crop_video)
st.video(crop_video)


