import streamlit as st
import cv2
import numpy as np
import operator
from string import ascii_uppercase
from keras.models import model_from_json
from hunspell import Hunspell
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from PIL import Image
import time

# Load Models
def load_models():
    # Main model
    json_file = open("Models1/model_new.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights("Models1/model_new.h5")

    # DRU model
    json_file_dru = open("Models1/model-bw_dru.json", "r")
    model_json_dru = json_file_dru.read()
    json_file_dru.close()
    loaded_model_dru = model_from_json(model_json_dru)
    loaded_model_dru.load_weights("Models1/model-bw_dru.h5")

    # TKDI model
    json_file_tkdi = open("Models1/model-bw_tkdi.json", "r")
    model_json_tkdi = json_file_tkdi.read()
    json_file_tkdi.close()
    loaded_model_tkdi = model_from_json(model_json_tkdi)
    loaded_model_tkdi.load_weights("Models1/model-bw_tkdi.h5")

    # SMN model
    json_file_smn = open("Models1/model-bw_smn.json", "r")
    model_json_smn = json_file_smn.read()
    json_file_smn.close()
    loaded_model_smn = model_from_json(model_json_smn)
    loaded_model_smn.load_weights("Models1/model-bw_smn.h5")

    return loaded_model, loaded_model_dru, loaded_model_tkdi, loaded_model_smn

# Video Processing Class
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.hs = Hunspell('en_US')
        self.ct = {char: 0 for char in ascii_uppercase}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.word = ""
        self.str = ""
        self.current_symbol = "Empty"
        self.loaded_model, self.loaded_model_dru, self.loaded_model_tkdi, self.loaded_model_smn = load_models()

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))
        
        prediction = {'blank': result[0][0]}
        for i, char in enumerate(ascii_uppercase):
            prediction[char] = result[0][i + 1]

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        # Secondary predictions
        if self.current_symbol in ['D', 'R', 'U']:
            prediction = {'D': result_dru[0][0], 'R': result_dru[0][1], 'U': result_dru[0][2]}
            self.current_symbol = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        elif self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {'D': result_tkdi[0][0], 'I': result_tkdi[0][1], 'K': result_tkdi[0][2], 'T': result_tkdi[0][3]}
            self.current_symbol = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)[0][0]
        elif self.current_symbol in ['M', 'N', 'S']:
            prediction1 = {'M': result_smn[0][0], 'N': result_smn[0][1], 'S': result_smn[0][2]}
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            if prediction1[0][0] == 'S':
                self.current_symbol = prediction1[0][0]

        if self.current_symbol == 'blank':
            for char in ascii_uppercase:
                self.ct[char] = 0
        else:
            self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 30:
            self.ct['blank'] = 0
            for char in ascii_uppercase:
                self.ct[char] = 0
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Define the bounding box coordinates
        x1 = int(0.5 * img.shape[1])
        y1 = 10
        x2 = img.shape[1] - 10
        y2 = int(0.5 * img.shape[1])
        
        # Draw the bounding box
        cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)

        # Preprocess the image for prediction
        cropped_img = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Perform prediction
        self.predict(res)
        
        # Display the predicted letter below the bounding box
        text_location = (x1 + 10, y2 + 30)  # Position the text below the bounding box
        cv2.putText(img, f"Predicted: {self.current_symbol}", text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Check if the current symbol has appeared for more than 30 frames
        if self.ct[self.current_symbol] > 30:
            if self.current_symbol != 'blank':
                self.word += self.current_symbol
                self.ct[self.current_symbol] = 0  # Reset the count for the current symbol

        # If the blank symbol is detected and there is a word formed, add the word to the sentence
        if self.current_symbol == 'blank' and self.blank_flag == 0:
            self.blank_flag = 1
            if len(self.word) > 0:
                self.str += " " + self.word  # Add word to sentence
                self.word = ""  # Reset the word
        elif self.current_symbol != 'blank':
            self.blank_flag = 0

        # Display the current word below the frame
        cv2.putText(img, f"Word: {self.word}", (x1 + 10, y2 + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
       

        # Return the annotated video frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")



# Run the Streamlit App
if __name__ == "__main__":
    st.title("Sign Language to Text Conversion")
    
    # Create placeholders for sentence and word
    sentence_placeholder = st.empty()
    word_placeholder = st.empty()

    webrtc_ctx = webrtc_streamer(key="sign-language", video_processor_factory=SignLanguageProcessor)
    
    if webrtc_ctx.video_processor:
        processor = webrtc_ctx.video_processor
        st.text("Streaming...")

        # Use placeholders to dynamically update the sentence and word
        while True:
            sentence_placeholder.text(f"Formed Sentence: {processor.str}")
            word_placeholder.text(f"Current Word: {processor.word}")
            
            # Update only when sentence or word changes
            time.sleep(1)  # Adjust the time for smoothness if necessary



