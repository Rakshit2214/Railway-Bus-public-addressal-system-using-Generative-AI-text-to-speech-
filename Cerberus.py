import streamlit as st
st. set_page_config(layout="wide")
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from bark.generation import generate_text_semantic
from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import imageio
import bark
import nltk
from bark import SAMPLE_RATE, generate_audio, preload_models, semantic_to_waveform
from IPython.display import Audio
import numpy as np
from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont
import imageio
from googletrans import Translator
import warnings
warnings.filterwarnings('ignore')
st.markdown("<h1 style='text-align: center; color: blue;'>Team Cerberus Project</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: skyblue;'>Automated Public Information Display System</h4>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: blue;'></h1>", unsafe_allow_html=True)

rad =st.sidebar.radio("Navigation",["Home"])

if(rad == 'Home'):
     if 'n' not in st.session_state:
          st.session_state.n = 0;
     if 'df' not in st.session_state:
          st.session_state.df = pd.DataFrame(columns=['CommandId','Author', 'Command'])
     if 'langlist' not in st.session_state:
          st.session_state.langlist = list()


     def GenrateVID(destlist, langualist):

          import re

          tags = []
          text = destlist[0][0]
          audpath=[]
          background_gif = ""
          # Check for relevant keywords
          if re.search(r'\btrain\b', text, re.I):
               tags.append("train")
               background_gif = 'BGGifs/extremely_detailed_india_817331524281062224_547018b6.mp4'
          if re.search(r'\bbus\b', text, re.I):
               tags.append("bus")
               background_gif = 'BGGifs/BUS_door_moving_real_li_1398796689843762855_d6a36105.mp4'
          if re.search(r'\bannoucement\b', text, re.I):
               tags.append("annoucement")
               background_gif = 'BGGifs/BUS_door_closing_door_o_6948728045410233927_a34c8c65.mp4'
          if re.search(r"\bdoor\b", text, re.I):
               tags.append("door")
               background_gif = 'BGGifs/Announcement_HD_megaph_4494000706456837908_42b71625.mp4'

          from gtts import gTTS
          from pydub import AudioSegment
          import soundfile as sf
          def download_audio(array, sample_rate, output_path):
               sf.write(output_path, array, sample_rate)
               audpath.append(output_path)

          cnto = 1

          def highQtts(sent_list, ln):
               global cnto
               cnto = 1
               if (ln == 'en'):
                    SPEAKER = "v2/en_speaker_7"

               else:
                    SPEAKER = "v2/hi_speaker_7"

               for i in sent_list:

                    sentences = nltk.sent_tokenize(i)
                    GEN_TEMP = 0.6
                    silence = np.zeros(int(0.10 * SAMPLE_RATE))  # quarter second of silence

                    pieces = []
                    for sentence in sentences:
                         semantic_tokens = generate_text_semantic(
                              sentence,
                              history_prompt=SPEAKER,
                              temp=GEN_TEMP,
                              min_eos_p=0.05,  # this controls how likely the generation is to end
                         )

                    audio_array = semantic_to_waveform(semantic_tokens, history_prompt=SPEAKER, )
                    pieces += [audio_array, silence.copy()]

                    audio_array = np.concatenate(pieces)
                    output_path = 'Audio/' + ln + 'speech' + '' + str(cnto) + '.mp3'
                    cnto += 1
                    # at[transcript]=(output_path,'english')

                    download_audio(audio_array, SAMPLE_RATE, output_path)

          cntt = 1

          def tts(text_list, ln):

               global cntt
               cntt = 1
               j = 0
               for i in text_list:
                    try:

                         ttss = gTTS(i, lang=ln)
                         path = 'Audio/' + ln + 'speech' + '' + str(cntt) + '.mp3'
                         ttss.save(path)
                         audpath.append(path)
                         cntt += 1
                         j += 1

                    except:
                         continue

          from moviepy.editor import VideoClip, AudioFileClip
          from PIL import Image, ImageDraw, ImageFont
          import imageio
          import numpy as np

          def generator(output_file, audio_file_path, background_gif_path, font_path, text, text_speed_factor=0.5):
               font_size = 40
               frame_rate = 30
               background_gif = imageio.get_reader(background_gif_path)
               audio = AudioFileClip(audio_file_path)
               audio_duration = audio.duration
               frame_width, frame_height = 640, 480  # Set the frame dimensions

               # Calculate the initial video duration
               initial_video_duration = audio_duration + 5  # Video duration is 5 seconds more than audio duration

               video_clip = VideoClip(make_frame=None, duration=initial_video_duration)

               # Define the font outside the if condition
               font = ImageFont.truetype(font_path, size=font_size)

               def make_frame(t):
                    # Get the current frame of the background GIF
                    frame_index = int(t * frame_rate) % len(background_gif)
                    background_frame = background_gif.get_data(frame_index)

                    # Create a frame with the background GIF frame
                    frame = Image.fromarray(background_frame)

                    # Calculate the text position based on time and video duration
                    x_position = frame_width - (frame_width + len(text) * font_size) * (
                                 t / (initial_video_duration / text_speed_factor))
                    y_position = (frame_height - font_size) / 4  # Adjust the text position

                    # Add a shadow effect
                    shadow_color = (255, 255, 0)  # Yellow shadow color
                    draw = ImageDraw.Draw(frame)
                    for offset in range(1, 5):  # Adjust the offset for a stronger or weaker shadow
                         shadow_position = (int(x_position) - offset, int(y_position) - offset)
                         draw.text(shadow_position, text, fill=shadow_color, font=font)

                    # Draw the main text
                    draw.text((int(x_position), int(y_position)), text, fill=(0, 0, 0), font=font)  # Black text color

                    return np.array(frame)

               # Set the video clip's make_frame function
               video_clip = video_clip.set_make_frame(make_frame)

               # Add audio to the video clip
               video_clip = video_clip.set_audio(audio)

               # Write the video with synchronized audio and text scrolling from right to left
               video_clip.write_videofile(output_file, codec="libx264", fps=frame_rate, audio_codec='aac')

               print(f"Video with audio and synchronized text scrolling from right to left generated: {output_file}")

          font_dict = {}
          font_dict['hi'] = 'Fonts/MANGAL.TTF'
          font_dict['en'] = 'Fonts/Myriad Pro Regular.ttf'
          font_dict['bn'] = 'Fonts/vrinda.ttf'
          font_dict['kn'] = 'Fonts/NotoSerifKannada-VariableFont_wght.ttf'
          font_dict['ml'] = 'Fonts/NotoSerifMalayalam-VariableFont_wght.ttf'
          font_dict['mr'] = 'Fonts/NotoSans-Black.ttf'
          font_dict['ta'] = 'Fonts/NotoSansTamil-VariableFont_wdth,wght.ttf'
          font_dict['te'] = 'Fonts/NotoSansTelugu-VariableFont_wdth,wght.ttf'
          font_dict['ur'] = 'Fonts/NotoNaskhArabic-VariableFont_wght.ttf'

          highQtts(destlist[0],langualist[0])
          highQtts(destlist[1],langualist[1])

          for i in range(2,len(langualist)):
               tts(destlist[i],langualist[i])
          kki=0
          for i in range(len(audpath)):
               generator('Vedio/output'+str(kki)+'.mp4', audpath[i], background_gif,
               font_dict[langualist[i]],destlist[i][0])


     def addtodf(a,b):
          st.session_state.n +=1;
          new_row = {'CommandId': st.session_state.n, 'Author': a, 'Command': b}
          st.session_state.df.loc[len(st.session_state.df)] = new_row


     def updatedf(num,au,co):
          condition = [num]
          a = st.session_state.df.loc[st.session_state.df['CommandId'].isin(condition)].Author
          st.session_state.df.Author[a.index[0]] = au
          b = st.session_state.df.loc[st.session_state.df['CommandId'].isin(condition)].Command
          st.session_state.df.Command[b.index[0]] = co

     def delfromdf(num):
          st.session_state.df.drop(st.session_state.df.index[(st.session_state.df['CommandId'] == num)],
                                   axis=0,inplace=True)


     def showdataframes(destlist,langlist,new_d) :
          for i in range(1, len(destlist)):
               # Dataframe
               newname = new_d[langlist[i]]
               st.write('For Language' ,newname)
               df_disp = st.session_state.df
               df_disp['Command'] = destlist[i]
               fig = go.Figure(data=[go.Table(
                    columnwidth=[200, 200, 700],
                    header=dict(values=['ID', 'Author', 'Commamnd'], fill_color='paleturquoise',
                                font=dict(color='black'), font_size=20),
                    cells=dict(values=[df_disp['CommandId'], df_disp['Author'], df_disp['Command']],
                               fill_color='lavender', font=dict(color='black'), font_size=14, height=20))
               ])
               st.plotly_chart(fig, use_container_width=True,height=200)
     def translatetolang(text, lang_list):
          translator = Translator()
          result = []
          for j in lang_list:
               bop = []
               for i in text:
                    translation = translator.translate(i, src='en', dest=str(j))
                    bop.append(translation.text)
               result.append(bop)
          return result

     # st.title("Real-Time Clock")
     # # Function to continuously update and display the current time
     # def display_real_time_clock():
     #      global st
     #      while True:
     #           current_time = time.strftime("%Y-%m-%d %H:%M:%S")
     #           time_display = st.empty()
     #           st.write(datetime.now().time())
     #           time.sleep(1)
     #
     #
     # time_display = st.empty()
     #
     # # Create a thread for the clock
     # clock_thread = threading.Thread(target=display_real_time_clock)
     # clock_thread.daemon = True  # Set the thread as a daemon so it exits when the main program exits
     #
     # # Start the clock thread
     # clock_thread.start()

     from datetime import datetime
     placeholder = st.empty()

     col1, col2, col3 = placeholder.columns([6, 1, 3])
     with col3:
          st.write(datetime.now().time())

     col1, col2, col3 = st.columns([6, 1, 3])
     with col1:
          st.write("Add Commands:")
          author = st.text_input('Command given By ', '')
          name = st.text_input('Give Command ', '')
          if st.button('Add',key=1):
               addtodf(author,name)

     with col3:
          st.write("Delete Commands:")
          id = st.text_input('Enter Id of the command to be deleted ', '')
          if st.button('Delete', key=2):
               delfromdf(int(id))

     # st.write("Update Command:")
     # id_u = st.text_input('Command Id of record to be updated')
     # author_u = st.text_input('Author ', '')
     # name_u = st.text_input('Command ', '')
     # if st.button('Update', key=3):
     #      updatedf(id_u,author_u, name_u)

     col1, col2, col3 = st.columns([1, 8, 1])
     with col2:
          # Dataframe
          st.markdown("<h2 style='text-align: left; color: white;'>Current List</h2>", unsafe_allow_html=True)
          df_disp = st.session_state.df
          df_disp.rename(columns={'ColumnId': 'Id', 'Author': 'Author', 'Command': 'Command'}, inplace=True)
          st.dataframe(df_disp, width=840, height=300)

     input_text = st.session_state.df['Command'].tolist()

     st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
     col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
     with col1:
          bn = st.checkbox("Bengali")
     with col2:
          kn = st.checkbox("Kannada")
     with col3:
          ml = st.checkbox("Malayalam")
     with col4:
          mr = st.checkbox("Marathi")


     c1, col2, col3, col4= st.columns([0.5, 1, 1, 1,])
     with col2:
          ta = st.checkbox("Tamil")
     with col3:
          te = st.checkbox("Telugu")
     with col4:
          ur = st.checkbox("Urdu")


     st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
     col1, col2, col3 = st.columns([4, 1, 4])
     with col1:
          if st.button('Translate', key=4):
               st.session_state.langlist.clear()
               st.session_state.langlist.append('en')
               st.session_state.langlist.append('hi')
               if bn:
                    st.session_state.langlist.append('bn')
               if kn:
                    st.session_state.langlist.append('kn')
               if ml:
                    st.session_state.langlist.append('ml')
               if mr:
                    st.session_state.langlist.append('mr')
               if ta:
                    st.session_state.langlist.append('ta')
               if te:
                    st.session_state.langlist.append('te')
               if ur:
                    st.session_state.langlist.append('ur')

               # st.write(st.session_state.langlist)
               # vidgeneration(input_text,st.session_state.langlist)

               # for i in l_list:
               #      if i in st.session_state.langlist:

               destlist = translatetolang(input_text,st.session_state.langlist)
               new_d = {'en': 'English',
                         'hi': 'Hindi',
                         'bn': 'Bengali',
                        'kn': 'Kannada',
                        'ml': 'Malayalam',
                        'mr': 'Marathi',
                        'ta': 'Tamil',
                        'te': 'Telugu',
                        'ur': 'Urdu'}

               showdataframes(destlist, st.session_state.langlist, new_d)

     st.markdown("<h1 style='text-align: left; color: white;</h1>", unsafe_allow_html=True)
     col1, col2, col3 = st.columns([4, 1, 4])
     with col1:
          if st.button('Publish', key=5):
               st.session_state.langlist.clear()
               st.session_state.langlist.append('en')
               st.session_state.langlist.append('hi')
               if bn:
                    st.session_state.langlist.append('bn')
               if kn:
                    st.session_state.langlist.append('kn')
               if ml:
                    st.session_state.langlist.append('ml')
               if mr:
                    st.session_state.langlist.append('mr')
               if ta:
                    st.session_state.langlist.append('ta')
               if te:
                    st.session_state.langlist.append('te')
               if ur:
                    st.session_state.langlist.append('ur')

               st.write(st.session_state.langlist)
               destlist = translatetolang(input_text, st.session_state.langlist)
               GenrateVID(destlist, st.session_state.langlist)
# favorite_command = edited_df.loc[edited_df["rating"].idxmax()]["command"]
# st.markdown(f"Your favorite command is **{favorite_command}** 🎈")
