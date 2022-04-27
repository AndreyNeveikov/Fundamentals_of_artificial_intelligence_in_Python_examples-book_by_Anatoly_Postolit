# Listing 7.24
def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
 print('СЕКУНДА : ', second_number)
 print('Массив выходных данных об объектах каждого кадра ', output_arrays)
 print('Массив уникальных объектов в кадрах: ', count_arrays)
 print('Среднее количество объектов в течение секунды: ', average_output_count)
 print('————КОНЕЦ СЕКУНДЫ ————–')

video_detector.detectObjectsFromVideo(input_file_path="holo1.mp4",
                      output_file_path=os.path.join(execution_path,
                      "holo1-detected"),
                      frames_per_second=30,
                      minimum_percentage_probability=40,
                      return_detected_frame(True),
                      per_frame_function(forSeconds),
                      log_progress=True)
