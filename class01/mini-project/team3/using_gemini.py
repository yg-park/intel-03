##### Gemini #####

import pathlib
import textwrap
import google.generativeai as genai

class Using_gemini():
    def __init__(self):
        self.GOOGLE_API_KEY = "AIzaSyAhfM6cvzTsGELKA1Wv8-Inc7W4UD2Dr2U"
        genai.configure(api_key=self.GOOGLE_API_KEY)
        self.model =  genai.GenerativeModel('gemini-pro')
        self.store_path = '../text/keyword.txt'
        self.diary_path = '../text/diary.txt'

    def get_keyword(self):
        with open(self.diary_path, 'r') as file:
            content = file.read()

        text = content + "\nTell me one of the musical atmospheres that suits this text.\
        and Extract keywords from this text.\
        separatly"
        
        #print(text)

        # Send
        response = self.model.generate_content(text)
        # Save the response text to a file
        with open(self.store_path, 'w', encoding='utf-8') as file:
            file.write(response.text)

##### End Gemini #####


