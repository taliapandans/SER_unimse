import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import azure.cognitiveservices.speech as speechsdk
import soundfile as sf

def extract_features(file_path, sr=16000, n_mels=64, fixed_length=64):
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)

    # Truncate or pad to get fixed shape
    if mel.shape[1] > fixed_length:
        mel = mel[:, :fixed_length]
    elif mel.shape[1] < fixed_length:
        padding = np.zeros((n_mels, fixed_length - mel.shape[1]))
        mel = np.hstack((mel, padding))

    return mel.T

    return mel
def transcribe_audio(filename, speech_config):
    audio_config = speechsdk.audio.AudioConfig(filename=filename)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized in {}".format(filename))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
    return None


def process_files(directory, speech_config):
    data = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory, file_name)
            cur_features = extract_features(file_path)
            transcript = transcribe_audio(file_path, speech_config)
            file_id, _, _, gender, label, sentiment, language, locale = file_name.split('+')

            # Convert sentiment to string representation
            sentiment_num = int(sentiment)
            sentiment_str = 'positive' if sentiment_num == 1 else 'negative' if sentiment_num == -1 else 'neutral'

          

            # Create the desired tuple structure
            data_tuple = (
                (
                    [],
                    [],
                    np.array(cur_features),
                    transcript,
                    0,
                    np.int32(len(cur_features))
                ),
                f"{sentiment_str},{sentiment_num},{label},{label[:3]}",
                file_id
            )
            data.append(data_tuple)
    return data

if __name__ == "__main__":
    speech_config = speechsdk.SpeechConfig(subscription="b75aa7c0b971460a811ae8785cdb9ff3", region="germanywestcentral")
    speech_config.speech_recognition_language = "de-DE"
    data = process_files("./preprocessed-emodb", speech_config)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_data, test_size=0.25, random_state=42)
    with open('emoDB_train_data.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open('emoDB_valid_data.pkl', 'wb') as f:
        pickle.dump(valid_data, f)
    with open('emoDB_test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    # Saving the entire data to a pickle
    with open('emoDB_all_data.pkl', 'wb') as f:
        pickle.dump(data, f)
