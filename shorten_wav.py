import soundfile as sf

# Read file
data, sample_rate = sf.read("speech_sample.wav")

# Keep only between 2s and 5s
t1 = 2
t2 = 5

start = int(t1 * sample_rate)
end   = int(t2 * sample_rate)

short_data = data[start:end]

# Save
sf.write("short.wav", short_data, sample_rate)