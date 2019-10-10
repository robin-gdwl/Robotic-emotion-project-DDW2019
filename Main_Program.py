import time

Robot = robotmotionclass()
face_finder = faceoperationclass()

Robot.move_home()

while True:
    while face_finder.findface == False:
        Robot.lookaraound()

    watch_time = time.time() + 5

    while True:
        if face_finder.findface == True:
            face_screen_location = Coordinateclass(face_finder.facelocation())
            face_real_location = face_screen_location.convert_coords
        else:
            break

        in_bounds = Robot.testmove()

        if time.time() < watch_time and in_bounds ==True:
            Robot.move(face_real_location)
            continue

        else:
            face_finder.landmark_detection()
            face_landmarks = face_finder.landmarks #should be a list of list of coordinates
            face_finder.detect_emotion()
            emotion_score = face_finder.emotion # list of strings with top 3 emotions

            Robot.move_to_write()
            Robot.draw_landmarks(face_landmarks)
            Robot.write_results(emotion_score)
            Robot.move_home

            break
