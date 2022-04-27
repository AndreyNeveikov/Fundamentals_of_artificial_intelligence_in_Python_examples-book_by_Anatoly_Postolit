Листинг 7.23
def forFrame(frame_number, output_array, output_count):
print('КАДР ' , frame_number)
print('Массив уникальных объектов в кадре: ', output_array)
print('Количество найденных объектов: ', output_count)
print('————КОНЕЦ КАДРА ————–')

video_detector.detectObjectsFromVideo(input_file_path="holo1.mp4",
                       output_file_path=os.path.join(execution_path,
                       "holo1-detected"),
                       frames_per_second=30,
                       minimum_percentage_probability=40,
                       per_frame_function(forFrame),
                       return_detected_frame(True),
                       log_progress=True)
