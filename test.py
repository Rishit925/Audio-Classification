from app.predict import predict

audio_path = "./data/Dataset/Yasser_Aldossary/yasser_noiseRed_127.wav"      # Replace with your test audio file

result = predict(audio_path)

print(result)