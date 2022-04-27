# Listing 7.15
def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("МИНУТА : ", minute_number)
    print("Массив выходных данных каждого кадра ", output_arrays)
    print("Массив количества уникальных объектов в каждом кадре: ", count_arrays)
    print("Среднее количество  объектов за минуту: ", average_output_count)
    print("------------КОНЕЦ МИНУТЫ --------------")


# Обращение к данной функции будет выглядеть следующим образом:

video_detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                      output_file_path=os.path.join(execution_path,
                                       "video_second_analysis"),
                                      frames_per_second=20,
                                      per_second_function=forMinute,
                                      minimum_percentage_probability=30,
                                      return_detected_frame=True,
                                      log_progress=True)
