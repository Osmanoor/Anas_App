import streamlit as st
import tempfile
import pipeline
import crop
import subprocess

def run_command(args):
    """Run command, transfer stdout/stderr back into Streamlit and manage error"""
    st.info(f"Running '{' '.join(args)}'")
    result = subprocess.run(args, capture_output=True, text=True)
    try:
        result.check_returncode()
        st.info(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(result.stderr)
        raise e

run_command("export CUDA_VISIBLE_DEVICES='0'")
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


