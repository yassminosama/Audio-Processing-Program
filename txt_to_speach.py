from tkinter import *
import customtkinter
from gtts import gTTS
from playsound import playsound
###############################################################################################################################
root = customtkinter.CTk()
root.geometry('485x170')
root.title('TEXT TO SPEECH')
customtkinter.set_appearance_mode("system")  # Modes: system (default), light, dark
customtkinter.set_default_color_theme("blue")  # Themes: blue (default), dark-blue, blue

#heading
customtkinter.CTkLabel(root, text = 'TEXT TO SPEECH' ).grid(row=0,columnspan=3,pady=20)

#label
customtkinter.CTkLabel(root, text ='Enter Text').grid(row=1,column=0)

#text variable
Msg = StringVar()

entry_field = customtkinter.CTkEntry(root,textvariable =Msg,width=190)
entry_field.grid(row=1,columnspan=3)

#Function to Convert Text to Speech in Python
def Text_to_speech():
    Message = entry_field.get()
    speech = gTTS(text = Message)
    speech.save('Text.mp3')
    playsound('Text.mp3')
    
#Function to Exit  
def Exit():
    root.destroy()

#Function to Reset  
def Reset():
    Msg.set("")

#Button
customtkinter.CTkButton(root, text = "PLAY" , command = Text_to_speech).grid(row=2,column=0,pady=20,padx=10)
customtkinter.CTkButton(root, text = 'RESET', command = Reset).grid(row=2,column=1,pady=20,padx=10)
customtkinter.CTkButton(root,text = 'EXIT' ,fg_color="#D35B58", hover_color="#C77C78",text_color="#ffffff", command = Exit).grid(row=2,column=2,pady=20,padx=10)
root.mainloop()